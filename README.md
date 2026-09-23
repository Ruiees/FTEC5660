# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.


## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution: 
> to students: please fill your solution description here.

Solution description

My agent follows a two-stage "extract-then-compute" design rather than asking the vision model to answer the two questions directly. In build_chain() I instantiate ChatDeepSeek with the deepseek-v4-flash-vision-exp model and a fixed system prompt that instructs the model to read each receipt image and output only three structured fields: SUBTOTAL, ROUNDING, and a comma-separated list of every discount line as positive numbers. In answer_queries() I convert each image to a base64 data URL via the provided helper, wrap it in a multimodal HumanMessage, invoke the chain once per receipt, and parse the reply with regular expressions. All arithmetic is then done deterministically in Python: total spent is the sum of SUBTOTAL + ROUNDING across receipts (the actual amount paid after rounding), and the no-discount total is the sum of SUBTOTAL + DISCOUNTS (discounts added back, rounding excluded). This separation matters because LLMs are reliable at reading text from images but error-prone at multi-step addition; by restricting the model's output to raw numbers and enforcing a single-amount response format (HK$...), I avoid both calculation mistakes and the grader's false-positive failures caused by stray numbers in the response. The prompt explicitly enumerates discount patterns (packaging-defect marks, "Buy X Save $ Y", percentage-off lines, coupon/member/voucher rows) and requires each occurrence to be listed separately so that repeated discounts are not collapsed into one.

Chain design
flowchart LR
    A[receipt images] --> B[image_data_url: base64 data URL]
    B --> C[HumanMessage with image_url block]
    D[SystemMessage<br/>extraction prompt] --> C
    C --> E[ChatDeepSeek<br/>deepseek-v4-flash-vision-exp]
    E --> F[parse: SUBTOTAL / ROUNDING / DISCOUNTS]
    F --> G[aggregate over all receipts]
    G --> H1["QUERY_1 = Σ(SUBTOTAL + ROUNDING)"]
    G --> H2["QUERY_2 = Σ(SUBTOTAL + DISCOUNTS)"]
    H1 & H2 --> I["results.csv"]