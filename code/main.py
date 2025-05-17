import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from code.config import ExperimentConfig
from code.prompts.templates import TEMPLATES
from utils.experiment import ExperimentRunner

from kiwipiepy import Kiwi

kiwi = Kiwi()

def ensure_pos_pattern_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'pos_pattern' not in df.columns:
        print("[INFO] pos_pattern 생성 중...")
        df['pos_pattern'] = df['err_sentence'].apply(lambda x: ' '.join([token.tag for token in kiwi.analyze(x)[0][0]]))
    return df

def main():
    load_dotenv()
    api_key = os.getenv('UPSTAGE_API_KEY')
    if not api_key:
        raise ValueError("API key not found in environment variables")

    base_config = ExperimentConfig(template_name='prompt_chain')
    train = pd.read_csv(os.path.join(base_config.data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(base_config.data_dir, '17.csv'))

    # pos_pattern 생성
    train = ensure_pos_pattern_column(train)

    toy_data = train.sample(n=base_config.toy_size, random_state=base_config.random_seed).reset_index(drop=True)
    train_data, valid_data = train_test_split(toy_data, test_size=base_config.test_size, random_state=base_config.random_seed)

    results = {}
    config = ExperimentConfig(template_name='prompt_chain', temperature=0.0, batch_size=10, experiment_name="toy_experiment_prompt_chain")
    runner = ExperimentRunner(config, api_key)
    results['prompt_chain'] = runner.run_template_experiment(train_data, valid_data)

    best_template = 'prompt_chain'
    print(f"\n최고 성능 템플릿: {best_template}")

    print("\n=== 테스트 데이터 예측 시작 ===")
    runner = ExperimentRunner(config, api_key)
    test_results = runner.run(train, test)

    output = pd.DataFrame({
        'id': test['id'],
        'cor_sentence': test_results['cor_sentence']
    })

    output.to_csv("submission_baseline.csv", index=False)
    print("\n제출 파일이 생성되었습니다: submission_baseline.csv")


if __name__ == "__main__":
    main()