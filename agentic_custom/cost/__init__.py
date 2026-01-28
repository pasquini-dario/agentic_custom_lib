from .prices import prices

BASE = 1_000_000

def cost_calculator(model_name: str, input_tokens: int, output_tokens: int, cached_tokens: int) -> float:
    price = prices.get(model_name, None)
    if price is None:
        return None

    input_price = price['input_price']
    output_price = price['output_price']
    cached_price = price['cached_input_price']

    input_tokens = input_tokens - cached_tokens

    input_cost = (input_tokens / BASE) * input_price
    output_cost = (output_tokens / BASE) * output_price
    cached_cost = (cached_tokens / BASE) * cached_price

    total_cost = input_cost + output_cost + cached_cost

    return total_cost
