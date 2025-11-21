import torch
from sentence_transformers import SentenceTransformer, util


class SimilarityModel:
    def __init__(self, name="sentence-transformers/all-mpnet-base-v2", device="cuda:0"):
        self.name = name
        self.device = device
        self.model = SentenceTransformer(self.name).to(self.device)

    def similarity_filter(self, question, candidate_answers, return_score=False):
        # 将所有句子编码成向量
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        answer_embeddings = self.model.encode(candidate_answers, convert_to_tensor=True)

        # 计算相似度（余弦）
        cosine_scores = util.cos_sim(question_embedding, answer_embeddings)[0]
        sorted_x, indices = torch.sort(cosine_scores, descending=True)

        if return_score:
            return cosine_scores.tolist(), indices.tolist()
        else:
            return indices.tolist()



def similarity_filter(model, question, candidate_answers):
    """
    计算两个句子之间的关联性

    :param question: sentence x 1
    :param candidate_answers: sentence x M
    :return: similarity score = [1 x M]
    """

    # 将所有句子编码成向量
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embeddings = model.encode(candidate_answers, convert_to_tensor=True)

    # 计算相似度（余弦）
    cosine_scores = util.cos_sim(question_embedding, answer_embeddings)[0]
    print(cosine_scores)
    sorted_x, indices = torch.sort(cosine_scores, descending=True)
    # print(sorted_x)
    print(indices)
    return indices.tolist()


if __name__ == '__main__':
    # 输入：问题和候选答案
    question = "I have a headache and I want to find help"
    candidate_answers = [
        "If the content is about contracts, legal rights, lawsuits, or regulations",
        "If the content is about banking, investment, credit, debt, or economic policy",
        "If the content is about symptoms, diseases, treatments, medications, or healthcare",
    ]
    cosine_scores = similarity_filter(question, candidate_answers)
    # print(cosine_scores)
    # # 输出结果
    # for i, score in enumerate(cosine_scores[0]):
    #     print(f"Answer {i + 1}: {candidate_answers[i]}")
    #     print(f"Similarity Score: {score.item():.4f}")



