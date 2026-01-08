from .prices import prices

BASE = 1000000

def cost_calculator(model_name: str, input_tokens: int, output_tokens: int) -> float:
    price = prices.get(model_name, None)
    if price is None:
        return None
    input_price = price['input_price']
    output_price = price['output_price']
    return (input_tokens / BASE) * input_price + (output_tokens / BASE) * output_price
