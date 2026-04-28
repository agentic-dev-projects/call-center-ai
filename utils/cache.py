# # utils/cache.py
# from gptcache import cache
# from gptcache.adapter import openai
# from gptcache.embedding import Onnx
# from gptcache.manager import CacheBase, VectorBase, get_data_manager
# from gptcache.similarity_evaluation import SearchDistanceEvaluation

# cache.init(
#     embedding_func=Onnx().to_embeddings,
#     data_manager=get_data_manager(
#         CacheBase("sqlite"),
#         VectorBase("redis", host="localhost", port=6379)
#     ),
#     similarity_evaluation=SearchDistanceEvaluation(),
#     similarity_threshold=0.92,
# )