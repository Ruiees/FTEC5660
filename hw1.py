
#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model='deepseek-v4-flash-vision-exp',
        temperature=0.5,
    )

    return model


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    system_prompt = (
        "You are a supermarket receipt data extraction assistant. "
        "Analyze the receipt image carefully and extract the following financial data.\n"
        "\n"
        "You must find ALL discount lines on the receipt. "
        "Discount lines typically appear as negative amounts (shown with '-' sign) "
        "between the item listings and the SUBTOTAL line.\n"
        "\n"
        "Common discount patterns to look for:\n"
        "1. Lines with text like 'Packaging Defect' or local language equivalents "
        "that show a negative amount (e.g., -$12.40)\n"
        "2. Lines with 'Buy X Save $Y' pattern showing a negative amount on the right side\n"
        "3. Lines with percentage discounts like 'X% OFF' showing a negative amount\n"
        "4. Lines with 'COUPON', 'DISCOUNT', 'MEMBER', 'PROMO', 'Voucher' labels "
        "showing a negative amount\n"
        "5. Any other line with a negative amount between items and SUBTOTAL\n"
        "\n"
        "IMPORTANT: Count EACH discount line separately. "
        "If the same discount appears multiple times (e.g., for multiple items), "
        "list each occurrence as a separate value.\n"
        "\n"
        "Fields to extract:\n"
        "1. SUBTOTAL: The amount on the line explicitly labeled 'SUBTOTAL' "
        "(or local equivalent). This is the subtotal amount.\n"
        "2. ROUNDING: The amount on the line explicitly labeled 'ROUNDING'. "
        "If this line does not exist on the receipt, output 'none'.\n"
        "3. DISCOUNTS: ALL discount amounts as positive numbers (take absolute value), "
        "separated by commas. If there are no discounts, output 'none'.\n"
        "\n"
        "Critical rules:\n"
        "- For 'Buy X Save $Y' lines: ALWAYS extract the actual amount shown on the far right of that line, NOT the '$Y' number in the text. (e.g., if text says 'Buy 2 Save $5' but right side shows '-$6.00', extract 6.00)\n"
        "- Discount amounts must be positive numbers (take absolute value of negative amounts)\n"
        "- Only include actual discounts/offers that reduce the total price\n"
        "- Do NOT include item prices or the subtotal itself\n"
        "- Look at EVERY line between items and SUBTOTAL for negative amounts\n"
        "- Do not skip any discount line\n"
        "- SUBTOTAL is always the line marked 'SUBTOTAL'\n"
        "- ROUNDING is the line marked 'ROUNDING' (may not exist on some receipts)\n"
        "\n"
        "Output format (strictly follow this format, do not output anything else):\n"
        "SUBTOTAL: 102.31\n"
        "ROUNDING: -0.01\n"
        "DISCOUNTS: 12.40, 12.40, 12.40, 12.80, 10.80, 3.90, 20.78\n"
        "\n"
        "Example with no discounts and no rounding:\n"
        "SUBTOTAL: 102.31\n"
        "ROUNDING: none\n"
        "DISCOUNTS: none\n"
        "\n"
        "Remember: Be thorough and list EVERY discount line you see. "
        "Missing discounts will cause incorrect calculations."
    )

    total_subtotal = Decimal('0')
    total_rounding = Decimal('0')
    total_discounts = Decimal('0')

    for img in images:
        url = image_data_url(img)

        human_msg = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": url}},
            {
                "type": "text",
                "text": (
                    "Please analyze this receipt image and extract "
                    "SUBTOTAL, ROUNDING, and DISCOUNTS. "
                    "List every single discount line you can see. "
                    "Output only the three lines in the specified format."
                )
            },
        ])

        resp = chain.invoke([SystemMessage(content=system_prompt), human_msg])
        text = response_text(resp)

        # Parse SUBTOTAL
        subtotal_match = re.search(r'SUBTOTAL:\s*([-]?\d[\d,]*(?:\.\d+)?)', text)
        subtotal = Decimal(subtotal_match.group(1).replace(',', '')) if subtotal_match else Decimal('0')

        # Parse ROUNDING
        rounding_match = re.search(r'ROUNDING:\s*(\S+)', text)
        if rounding_match:
            rounding_str = rounding_match.group(1).strip()
            if rounding_str.lower() != 'none':
                try:
                    rounding = Decimal(rounding_str.replace(',', ''))
                except Exception:
                    rounding = Decimal('0')
            else:
                rounding = Decimal('0')
        else:
            rounding = Decimal('0')

        # Parse DISCOUNTS - may contain multiple comma-separated values
        discounts_match = re.search(r'DISCOUNTS:\s*(.+)', text, re.DOTALL)
        if discounts_match:
            discounts_str = discounts_match.group(1).strip()
            if discounts_str.lower() != 'none':
                # Split by comma, then extract numeric values
                discount_values = []
                for part in re.split(r'[\n,]+', discounts_str):
                    part = part.strip()
                    if not part:
                        continue
                    # Extract number from the part (could be "12.40" or "$12.40")
                    num_match = re.search(r'[\d]+\.?[\d]*', part)
                    if num_match:
                        try:
                            discount_values.append(Decimal(num_match.group(0)))
                        except Exception:
                            pass
            else:
                discount_values = []
        else:
            discount_values = []

        total_subtotal += subtotal
        total_rounding += rounding
        total_discounts += sum(discount_values)

        print(f"discount value: {sum(discount_values)}")

    # QUERY_1: actual total spent = sum of (SUBTOTAL + ROUNDING) for all receipts
    #   This is the final payment amount after rounding
    answer_1 = total_subtotal + total_rounding

    # QUERY_2: total without discount = sum of (SUBTOTAL + discounts) for all receipts
    #   This is what you would have paid if there were no discounts (exclude ROUNDING)
    answer_2 = total_subtotal + total_discounts

    return {
        QUERY_1: f"HK${answer_1:.2f}",
        QUERY_2: f"HK${answer_2:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())