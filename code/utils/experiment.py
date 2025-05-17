import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import requests

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from code.utils.metrics import evaluate_correction

from kiwipiepy import Kiwi
from difflib import SequenceMatcher

kiwi = Kiwi()

# 품사 시퀀스 추출 함수
def get_pos_pattern(sentence):
    """
    입력된 문장에서 품사 시퀀스를 추출하여 문자열로 반환
    """
    tokens = kiwi.analyze(sentence)[0][0]
    return ' '.join([token.tag for token in tokens])

def get_structural_similarity(s1, s2):
    """
    두 문장의 품사 시퀀스 유사도를 반환 (0~1 사이)
    """
    pattern1 = get_pos_pattern(s1)
    pattern2 = get_pos_pattern(s2)
    return SequenceMatcher(None, pattern1, pattern2).ratio()

def find_top_k_similar_cached(input_sentence, cached_df, k=3):
    """
    캐싱된 품사 시퀀스를 활용한 빠른 Top-K 유사도 검색
    """
    input_pattern = get_pos_pattern(input_sentence)
    results = []
    for idx, row in cached_df.iterrows():
        similarity = SequenceMatcher(None, input_pattern, row['pos_pattern']).ratio()
        results.append((similarity, row['err_sentence'], row['cor_sentence']))

    results = sorted(results, key=lambda x: x[0], reverse=True)[:k]
    return results


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        self.template = TEMPLATES[config.template_name]
        self.api_url = config.api_url
        self.model = config.model
    
    def _make_prompt(self, text: str) -> str:
        """프롬프트 생성"""
        return self.template.format(text=text)
    
    def _call_api_single(self, prompt: str) -> str:
        """단일 문장에 대한 API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        return results["choices"][0]["message"]["content"]

    def run(self, df_train, df_test) -> pd.DataFrame:
        results = []
        prompt_chain = self.template  # Dict[str, str]

        for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
            origin = row['err_sentence']

            # 유사한 예시 5개 추출
            top_k_results = find_top_k_similar_cached(origin, df_train, k=3)
            few_shot_examples = ""
            for similarity, err, cor in top_k_results:
                few_shot_examples += f"입력: {err}\n출력: {cor}\n\n"
            few_shot_examples += f"입력: {origin}\n출력:"

            # 1단계 프롬프트
            prompt_1 = prompt_chain['prompt_1'].format(few_shot=few_shot_examples, origin=origin)
            cor_1 = self._call_api_single(prompt_1)

            # 2단계 프롬프트
            prompt_2 = prompt_chain['prompt_2'].format(origin=origin, correction=cor_1)
            result = self._call_api_single(prompt_2)

            results.append({
                'id': row['id'],
                'cor_sentence': result
            })

        return pd.DataFrame(results)

    def run_template_experiment(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> Dict:
        print(f"\n=== {self.config.template_name} 템플릿 실험 ===")
        print("\n[학습 데이터 실험]")
        train_results = self.run(train_data, train_data)
        train_recall = evaluate_correction(train_data, train_results)

        print("\n[검증 데이터 실험]")
        valid_results = self.run(train_data, valid_data)
        valid_recall = evaluate_correction(valid_data, valid_results)

        return {
            'train_recall': train_recall,
            'valid_recall': valid_recall,
            'train_results': train_results,
            'valid_results': valid_results
        }