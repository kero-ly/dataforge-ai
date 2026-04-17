from dataforge.strategies.base import BaseStrategy
from dataforge.strategies.evol_instruct import EvolInstruct
from dataforge.strategies.paraphrase import Paraphrase
from dataforge.strategies.seed_to_qa import SeedToQA
from dataforge.strategies.self_play import SelfPlay

__all__ = ["BaseStrategy", "EvolInstruct", "Paraphrase", "SeedToQA", "SelfPlay"]
