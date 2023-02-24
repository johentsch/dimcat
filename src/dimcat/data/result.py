# from typing import Optional, Tuple, Union
#
# import pandas as pd
#
#
#
#
#
# class NotesResult(Result):
#     def _concat_results(
#         self,
#         index_result_dict: Optional[dict] = None,
#         level_names: Optional[Union[Tuple[str], str]] = None,
#     ) -> pd.DataFrame:
#         df = super()._concat_results(
#             index_result_dict=index_result_dict, level_names=level_names
#         )
#         return df[sorted(df.columns)]
#
#
# class ChordSymbolResult(Result):
#     def _concat_results(
#         self,
#         index_result_dict: Optional[dict] = None,
#         level_names: Optional[Union[Tuple[str], str]] = None,
#     ) -> pd.DataFrame:
#         if index_result_dict is None:
#             index_result_dict = self.result_dict
#             if level_names is None:
#                 level_names = self.dataset_after.index_levels["indices"]
#         elif level_names is None:
#             raise ValueError("Names of index level(s) need(s) to be specified.")
#         df = pd.DataFrame.from_dict(index_result_dict, orient="index", dtype="Int64")
#         df.fillna(0, inplace=True)
#         try:
#             df.index.rename(level_names, inplace=True)
#         except TypeError:
#             print(f"level_names = {level_names}; nlevels = {df.index.nlevels}")
#             raise
#         # df.sort_values(df.index.to_list(), axis=1, ascending=False, inplace=True)
#         return df
#
#     def iter_group_results(self):
#         for idx, aggregated in super().iter_group_results():
#             yield idx, aggregated.sort_values(ascending=False).astype("Int64")
