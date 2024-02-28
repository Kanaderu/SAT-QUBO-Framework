import os
import argparse
import json

import pandas as pd
import numpy as np

from sat.models.prototypes.nmprototypesat import NMPrototypeSAT
from sat.qubosearcher import *
from sat.formula_creator import *

from sat.models.choi import ChoiSAT
from sat.models.chancellor import ChancellorSAT
from sat.models.nuesslein import NuessleinNM


def format_qubo_outputs(qubo):
   qubo_size = max(np.array(list(qubo.keys())[-1]) + 1)

   # qubo matrix
   M = np.zeros(shape=(qubo_size, qubo_size), dtype=np.int32)
   M[tuple(zip(*qubo.keys()))] = list(qubo.values())
   
   return M

def save_qubo_matrix(qubo, output, name):
   # convert from dok to square matrix
   qubo_matrix = format_qubo_outputs(qubo)

   output_params = {
      f"Output {name} CSV": os.path.join(output, name + '.csv'),
      f"Output {name} Numpy": os.path.join(output, name + '.npy'),
      f"Output {name} JSON": os.path.join(output, name + '.json'),
   }
   # save csv
   pd.DataFrame(qubo_matrix).to_csv(output_params[f'Output {name} CSV'], index=False, header=False)

   # save numpy array
   np.save(output_params[f'Output {name} Numpy'], qubo_matrix)

   # save json dok output
   with open(output_params[f'Output {name} JSON'], 'w') as f:
      qubo_json_encode_dict = {str(k): v for k, v in qubo.items()}
      json.dump(qubo_json_encode_dict, f)

   return output_params

def parse_pattern_qubo_args(args):
   if not args.mn_pattern_qubos or len(args.mn_pattern_qubos) != 3:
      print("Please provide min and max value and step size for finding all pattern QUBOs")
      exit(1)
   min_val = int(args.mn_pattern_qubos[0])
   max_val = int(args.mn_pattern_qubos[1])
   step_size = int(args.mn_pattern_qubos[2])
   return min_val, max_val, step_size

def search_pattern_qubos(min_val, max_val, step_size):
   print(f"Search all pattern QUBOs - (see https://arxiv.org/pdf/2305.02659.pdf)")
   find_all_mn_pattern_qubos(min_val, max_val, step_size)
   found_pattern_qubos = load_all_pattern_qubos(min_val, max_val, step_size)
   return found_pattern_qubos

if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='SAT QUBO Transformation')
   parser.add_argument('-d', '--dimacs', type=str, help='Path to DIMACS file')
   parser.add_argument('-o', '--output', type=str, help='Path to output folder')
   parser.add_argument('-t', '--transformation', type=str, choices=[
      'choi', 'chancellor', 'nuesslein', 'nm-prototype', "search-all-pattern-qubos"
   ], help='Transformation type: choi, chancellor, nuesslein, nm-prototype, search-all-pattern-qubos', default='choi')
   parser.add_argument('-p', '--mn-pattern-qubos', nargs='+', help='Find all pattern QUBOs for clause types 0-3 with min and max value and step size (e.g. -1 1 10)')
   args = parser.parse_args()

   if args.output:
      os.makedirs(args.output, exist_ok=True)

   params = {
      "Dimacs File: ": args.dimacs,
      "Output Folder: ": args.output,
      "Transformation: ": args.transformation,
   }
   if args.mn_pattern_qubos:
      params.update({"Min-Max-Step Pattern: ": args.mn_pattern_qubos})

   if args.dimacs:
      # load formula from dimacs file
      formula = load_formula_from_dimacs_file(args.dimacs)

      # create QUBO transformation
      if args.transformation == "choi":
         choi_sat = ChoiSAT(formula)
         choi_sat.create_qubo()

         if args.output:
            params.update(save_qubo_matrix(choi_sat.qubo, args.output, "choi"))

      elif args.transformation == "chancellor":
         chancellor_sat = ChancellorSAT(formula)
         chancellor_sat.create_qubo()

         if args.output:
            params.update(save_qubo_matrix(chancellor_sat.qubo, args.output, "chancellor"))
         
      elif args.transformation == "nuesslein":
         nuesslein_sat = NuessleinNM(formula)
         nuesslein_sat.create_qubo()

         if args.output:
            params.update(save_qubo_matrix(nuesslein_sat.qubo, args.output, "nuesslein"))

      elif args.transformation == "nm-prototype":
         found_pattern_qubos = search_pattern_qubos(*parse_pattern_qubo_args(args))
         # print(f'Found pattern QUBOs: {found_pattern_qubos}')

         min_num_pattern_sets = min([len(v) for v in found_pattern_qubos.values()])
         print(f"Build concrete QUBO-transformation from pattern qubos")
         for idx in range(min_num_pattern_sets):
            sat_transformation = NMPrototypeSAT(formula)
            sat_transformation.add_clause_qubos(found_pattern_qubos[0][idx], found_pattern_qubos[1][idx],
                                                found_pattern_qubos[2][idx], found_pattern_qubos[3][idx])
            sat_transformation.create_qubo()

            if args.output:
               params.update(save_qubo_matrix(sat_transformation.qubo, args.output, f"nm-prototype-{idx}"))

   # search all pattern qubos without dimacs input file
   if args.transformation == "search-all-pattern-qubos":
      found_pattern_qubos = search_pattern_qubos(*parse_pattern_qubo_args(args))
      for k, v in found_pattern_qubos.items():
         for idx, qubo_v in enumerate(v):
            if args.output:
               params.update(
                  save_qubo_matrix(qubo_v, args.output, f'found_pattern_qubos_clause_type_{k}-{idx}')
               )
   
   param_info = pd.Series(params).to_markdown(
      headers=["Parameter", "Value"],
      tablefmt="rounded_outline",
      numalign="left",
      stralign="left"
   )

   print(f"{param_info}\n")