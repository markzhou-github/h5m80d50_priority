import itertools
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from generate_signals import assign_priorities, record_history


class RulesTest(unittest.TestCase):
    def test_all_gate_combinations(self):
        frame = pd.DataFrame({'ts_code': ['a', 'b', 'c']})
        for k50, k60, market in itertools.product((False, True), repeat=3):
            if k50 and not k60:
                continue
            for top1 in ('a', 'outside'):
                result = assign_priorities(frame, top1, k50, k60, market)
                universe = set(frame.ts_code)
                current = universe - {top1} if market else set()
                p1 = current & (universe if k50 else set())
                p2 = ((universe if k60 else set()) | current) - p1
                self.assertEqual(set(result.loc[result.priority.eq(1), 'ts_code']), p1)
                self.assertEqual(set(result.loc[result.priority.eq(2), 'ts_code']), p2)
                self.assertFalse(p1 & p2)

    def test_history_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'history.csv'
            history = pd.DataFrame({'trade_date': ['20260901'], 'bad_risk_logit': [.4]})
            history = record_history(history, '20260902', .5, path)
            history = record_history(history, '20260902', .5, path)
            self.assertEqual(len(history), 2)
            self.assertEqual(history.loc[history.trade_date.lt('20260902'), 'bad_risk_logit'].tolist(), [.4])
            with self.assertRaises(ValueError):
                record_history(history, '20260902', .7, path)


if __name__ == '__main__':
    unittest.main()
