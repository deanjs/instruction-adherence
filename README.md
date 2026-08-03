# 코딩 에이전트의 지침 준수율 하락에서 선행 코드 패턴의 인과적 역할

건국대학교 ELION LAB

## 구조

```
src/         실험 스크립트
notebooks/   Colab 진입점
results/     실행 결과 (git에 커밋한다 — 사전 등록 기록으로 쓴다)
```

## Step 0 — 함수명 pair 토큰 수 일치 수율

3단계 개입 실험은 준수/위반 두 조건의 토큰 위치가 정확히 일치해야만 성립한다.
`getUserData` / `get_user_data` 가 같은 토큰 수로 쪼개지는 비율을 먼저 측정한다.

```bash
pip install -r requirements.txt
python src/step0_pair_yield.py
```

Colab: `notebooks/step0_colab.ipynb` 를 GitHub 탭에서 열어 실행.

**판정 기준**

| 일치율 | 판단 |
|---|---|
| 40% 이상 | 그대로 진행 |
| 20 ~ 40% | 이름 풀 확장으로 대응 가능 |
| 20% 미만 | 개입 대상 규칙을 표기 규약 다른 종류로 교체 검토 |
