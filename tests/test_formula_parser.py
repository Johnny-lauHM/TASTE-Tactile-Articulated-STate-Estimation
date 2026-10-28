"""Validate the actual parser without importing Isaac's package registration."""
import ast
from pathlib import Path
import re
import unittest
path=Path(__file__).resolve().parents[1]/'source/rubik_lab/rubik_lab/moves.py'
node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='parse_formula')
ns={'re':re}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns)
parse=ns['parse_formula']

class FormulaParserTests(unittest.TestCase):
 def test_standard_and_unicode_moves(self):
  self.assertEqual(parse("R U R’ U′ U2'"),['R','U',"R'","U'",'U2'])
 def test_repeated_nested_groups(self):
  self.assertEqual(parse("(R (U U')2 R')2"),['R','U',"U'",'U',"U'","R'"]*2)
 def test_adjacent_moves_and_whitespace(self):
  self.assertEqual(parse("RU2\nR'\t x2"),['R','U2',"R'",'x2'])
 def test_invalid_text_is_never_silently_discarded(self):
  for formula in ('U garbage', 'U3', "U''", 'U, R', 'y U', '(U', 'U)', '()','(U)0','U [R]'):
   with self.subTest(formula=formula),self.assertRaises(ValueError):parse(formula)
 def test_empty_is_no_op(self):
  self.assertEqual(parse('  '),[])
