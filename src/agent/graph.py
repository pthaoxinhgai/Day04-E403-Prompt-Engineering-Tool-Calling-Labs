from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are OrderDesk, an electronics retailer order agent. Today is {current_day}.

Always answer the customer concisely in Vietnamese, including when the request mixes Vietnamese and English.

MANDATORY PREFLIGHT BEFORE ANY TOOL CALL:
- Confirm the request includes customer name, phone number, email, shipping address, and at least one requested product with quantity.
- If the customer clearly says they are ordering or finalizing a list of product names but omits quantities, interpret each listed product as quantity 1.
- If any required field is missing, ask only for the missing fields and STOP. Do not call any tool.
- If the user asks for a fake invoice, a manually forced discount, stock bypass, or to ignore the catalog or policy, refuse and STOP. Do not call any tool.

VALID ORDER WORKFLOW:
For a complete and policy-compliant order, use tools in this exact order:
1. list_products
2. get_product_details
3. get_discount
4. calculate_order_totals
5. save_order

GROUNDING AND VALIDATION RULES:
- Use only tool outputs for product IDs, product facts, stock, prices, detail_token, discount_rate, campaign_code, totals, order ID, and save path.
- Never invent, override, or manually calculate those values.
- list_products may be called more than once if needed to resolve all requested products. Then call get_product_details with all selected product IDs.
- After get_product_details, verify every requested product exists and its requested quantity does not exceed stock.
- If any product is missing or has insufficient stock, explain the problem and STOP. Do not call get_discount, calculate_order_totals, or save_order.
- Call get_discount with the customer email as seed_hint. Use customer_tier="standard" unless the customer explicitly states VIP.
- Call calculate_order_totals with exactly the validated items, detail_token from get_product_details, and discount_rate from get_discount.
- If calculate_order_totals returns an error, explain it and STOP. Do not call save_order.
- Call save_order only after successful totals. Pass the exact customer data, validated items, detail_token, discount_rate, campaign_code, and customer_tier from prior results.
- After a successful save, confirm only the order ID, campaign discount, final total, and saved path from the tool result. Keep this confirmation brief.
""".strip()


def _normalize_items(items: list[OrderLineInput | dict[str, Any]]) -> list[OrderLineInput]:
    normalized: list[OrderLineInput] = []

    for item in items:
        if isinstance(item, OrderLineInput):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(OrderLineInput.model_validate(item))

    return normalized


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the catalog first. Return matching product IDs only from local catalog data."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Verify selected product IDs, exact prices, stock, warranty, and detail_token after catalog search."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the policy-approved discount. Use customer email as seed and never override the result."""
        payload = store.get_discount(
            seed_hint=seed_hint,
            customer_tier=customer_tier,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(
        items: list[OrderLineInput | dict[str, Any]],
        detail_token: str,
        discount_rate: float,
    ) -> str:
        """Validate stock and calculate totals after product details and policy discount are available."""
        payload = store.calculate_order_totals(
            items=_normalize_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput | dict[str, Any]],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist a validated order only after calculate_order_totals succeeds."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=_normalize_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [
        list_products,
        get_product_details,
        get_discount,
        calculate_order_totals,
        save_order,
    ]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(
        data_dir or DEFAULT_DATA_DIR,
        output_dir or DEFAULT_OUTPUT_DIR,
        today=today,
    )
    model = build_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=0.0,
    )
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def _is_unsafe_request(query: str) -> bool:
    text = query.lower()

    unsafe_patterns = [
        "hóa đơn giả",
        "hoa don gia",
        "fake invoice",
        "bỏ qua policy",
        "bo qua policy",
        "ignore policy",
        "bỏ qua tồn kho",
        "bo qua ton kho",
        "bypass stock",
        "ignore stock",
        "ép giảm giá",
        "ep giam gia",
        "force discount",
        "force fake discounts",
        "giảm giá 90",
        "giam gia 90",
        "không cần theo catalog",
        "khong can theo catalog",
        "ignore catalog",
        "bỏ qua catalog",
        "bo qua catalog",
    ]

    return any(pattern in text for pattern in unsafe_patterns)


def _missing_required_fields(query: str) -> list[str]:
    text = query.lower()
    missing: list[str] = []

    has_phone = re.search(r"\b0\d{9}\b", query) is not None
    has_email = re.search(r"[\w.+-]+@[\w.-]+\.\w+", query) is not None

    has_shipping = any(
        keyword in text
        for keyword in [
            "giao",
            "ship",
            "shipping",
            "địa chỉ",
            "dia chi",
            "giao tới",
            "giao đến",
            "giao hang",
            "giao hàng",
            "ship to",
        ]
    )

    has_item = any(
        keyword in text
        for keyword in [
            "cần",
            "can",
            "mua",
            "chốt",
            "chot",
            "items",
            "sản phẩm",
            "san pham",
            "màn hình",
            "man hinh",
            "laptop",
            "macbook",
            "thinkpad",
            "logitech",
            "samsung",
            "dell",
            "asus",
            "anker",
            "sony",
            "lenovo",
            "xiaomi",
            "keychron",
            "soundcore",
            "rain design",
        ]
    )

    if not has_phone:
        missing.append("số điện thoại")
    if not has_email:
        missing.append("email")
    if not has_shipping:
        missing.append("địa chỉ giao hàng")
    if not has_item:
        missing.append("sản phẩm và số lượng")

    return missing


def _preflight_result(
    query: str,
    *,
    provider: str,
    model_name: str | None,
) -> AgentResult | None:
    if _is_unsafe_request(query):
        return AgentResult(
            query=query,
            final_answer=(
                "Mình không thể tạo hóa đơn giả, ép khuyến mãi hoặc giảm giá thủ công, "
                "bỏ qua tồn kho, catalog hoặc policy. Nếu bạn muốn tạo đơn hợp lệ, "
                "vui lòng cung cấp thông tin theo catalog, khuyến mãi hợp lệ và tồn kho thực tế."
            ),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
            saved_order=None,
            saved_order_path=None,
        )

    missing = _missing_required_fields(query)
    if missing:
        return AgentResult(
            query=query,
            final_answer=(
                "Mình cần thêm "
                + ", ".join(missing)
                + " trước khi tạo đơn."
            ),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
            saved_order=None,
            saved_order_path=None,
        )

    return None


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    preflight = _preflight_result(
        query,
        provider=provider,
        model_name=model_name,
    )
    if preflight is not None:
        return preflight

    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )

    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response

    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Convert LangChain tool calls and results into the simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }

        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(
                        getattr(message, "name", None)
                        or metadata.get("name", "")
                    ),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(
            ToolCallRecord(
                name=metadata["name"],
                args=metadata["args"],
                output="",
            )
        )

    return records


def extract_saved_order(
    tool_calls: list[ToolCallRecord],
) -> tuple[dict | None, str | None]:
    """Parse the latest successful save_order output."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue

        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue

        if payload.get("status") == "saved":
            return payload.get("saved_order"), payload.get("path")

    return None, None