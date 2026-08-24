# CausalMind (Project Amygdala v5.7)
**A Biologically Inspired LLM with Metabolic Gating**

[Read in Korean (한국어 번역)](#-한국어-번역-korean-translation)

## Table of Contents
1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [The Problem: Limitations of Existing LLMs](#2-the-problem-limitations-of-existing-llms)
3. [Core Architecture: The Amygdala Solution](#3-core-architecture-the-amygdala-solution)
4. [Experimental Results & Debugging Journey](#4-experimental-results--debugging-journey)
5. [Future Direction: Amygdala vs. BitNet b1.58](#5-future-direction-amygdala-vs-bitnet-b158)

---

## 1. Project Overview & Motivation
Project Amygdala is an experimental 350M-parameter Large Language Model (LLM) designed to mimic the energy efficiency of a biological brain. Instead of performing uniform, heavy computations for every token, Amygdala dynamically allocates computational resources based on "Cognitive Surprisal" (how unexpected a token is).

During training up to 4,000 steps, the model encountered severe challenges like "Semantic Collapse" and "Semantic Looping." By deeply analyzing the model's internal states, we successfully debugged these issues without scaling up parameters, resulting in the highly optimized **Amygdala v5.7**.

---

## 2. The Problem: Limitations of Existing LLMs
1.  **Metabolic Inefficiency**: Traditional Transformers perform dense, full-depth forward passes for every token, regardless of predictability. This wastes immense computational energy on trivial syntax.
2.  **Semantic Looping & Positional Blindness**: Recurrent architectures often lose positional context over long contexts, trapping the model in endless repetition loops (e.g., repeating the same phrases or dates).
3.  **VRAM Bottlenecks (OOM)**: Scaling depth in parallel architectures drastically increases memory consumption, making them difficult to train on standard hardware (e.g., A100 40GB).

---

## 3. Core Architecture: The Amygdala Solution
*   **Metabolic Gating (Endocrine Gates)**: The network features a "Metabolic Tank." The model conditionally gates information flow based on the current tank level and the surprisal of the token, saving energy on predictable tokens.
*   **O(1) State Memory**: Utilizes `ChunkedParallelRecurrence` and `FastSurprisal` modules to maintain a constant memory footprint during sequence processing, avoiding the quadratic complexity of standard attention.
*   **Rotary Position Embedding (RoPE)**: Injected into the hybrid attention blocks in v5.7 to give the model absolute and relative coordinate awareness, permanently breaking semantic loops.

---

## 4. Experimental Results & Debugging Journey

### A. The "Token Salad" Phenomenon (Representation Collapse)
*   **Issue**: At 4,000 steps, the 350M model began outputting gibberish (`-alskman`, `omingiest`).
*   **Diagnosis (Tank Mismatch)**: The model learned to survive on a severely depleted metabolic tank ($\sim 0.0135$) during training due to a strict metabolic penalty. During inference, the tank was initialized at $1.0$ (100%), causing an "Activation Explosion" that shattered the softmax distribution.
*   **Fix**: Hardcoded the initial inference tank to match the stable training state (`init_tank=0.0135`), instantly restoring fluent English generation.

### B. Fixing OOM with Gradient Accumulation
*   **Issue**: The 350M model's 24-layer depth caused Out-Of-Memory (OOM) errors on a 40GB A100.
*   **Fix**: Reduced batch size from 32 to 8, and increased `grad_accum` to 32. This successfully dropped VRAM usage to $\sim 25GB$ while preserving the effective batch size.

---

## 5. Future Direction: Amygdala vs. BitNet b1.58
While Microsoft's **BitNet b1.58** achieves efficiency by quantizing weights (Precision Diet: -1, 0, 1) to eliminate multiplication, **Amygdala v5.7** achieves efficiency through **Metabolic Gating** (filtering computation based on surprisal). 

The ultimate goal of this project is to combine both: utilizing ternary weights for mathematical efficiency alongside metabolic gating for biological efficiency, creating the ultimate low-energy, highly capable cognitive model.

<br>
<br>

---

# 🇰🇷 한국어 번역 (Korean Translation)

# CausalMind (Project Amygdala v5.7)
**생물학적 에너지 게이팅을 모사한 초효율 LLM**

## 목차
1. [프로젝트 개요 및 동기](#1-프로젝트-개요-및-동기)
2. [기존 LLM의 한계와 문제점](#2-기존-llm의-한계와-문제점)
3. [핵심 아키텍처: Amygdala의 해결책](#3-핵심-아키텍처-amygdala의-해결책)
4. [실험 결과 및 디버깅 여정](#4-실험-결과-및-디버깅-여정)
5. [향후 방향: Amygdala vs BitNet b1.58](#5-향후-방향-amygdala-vs-bitnet-b158)

---

## 1. 프로젝트 개요 및 동기
Project Amygdala는 생물학적 뇌의 에너지 효율성을 모사하도록 설계된 350M 파라미터 규모의 실험적 대형 언어 모델(LLM)입니다. 모든 토큰에 대해 동일하고 무거운 연산을 수행하는 대신, "인지적 놀람(Cognitive Surprisal)"에 따라 연산 자원을 동적으로 배분합니다.

4,000 스텝의 학습 과정에서 모델은 "표상 붕괴(Semantic Collapse)"와 "의미론적 갇힘(Semantic Looping)"이라는 심각한 문제에 직면했습니다. 우리는 파라미터를 무작정 늘리는 대신 모델의 내부 상태를 심층적으로 분석하여 이 문제들을 디버깅해 냈으며, 그 결과 고도로 최적화된 **Amygdala v5.7**을 완성했습니다.

---

## 2. 기존 LLM의 한계와 문제점
1.  **에너지(대사) 비효율성**: 기존 트랜스포머는 예측하기 쉬운 뻔한 문법적 토큰에도 항상 모델 전체 깊이의 무거운 연산을 수행하여 막대한 컴퓨팅 에너지를 낭비합니다.
2.  **의미론적 갇힘 및 위치 감각 상실**: 순환(Recurrent) 아키텍처는 긴 문맥에서 위치 정보를 잃어버리기 쉬우며, 이로 인해 똑같은 구문이나 날짜를 무한히 반복하는 루프에 빠지게 됩니다.
3.  **VRAM 병목 현상 (OOM)**: 병렬 아키텍처에서 층수(Depth)를 늘리면 메모리 소비량이 급증하여, 일반적인 하드웨어(예: A100 40GB)에서는 훈련이 불가능해집니다.

---

## 3. 핵심 아키텍처: Amygdala의 해결책
*   **대사 게이팅 (Endocrine Gates)**: 네트워크 내부에 "대사 탱크(Metabolic Tank)"를 두어, 현재 탱크 잔량과 토큰의 놀람 수치에 따라 정보 흐름을 조건부로 차단/개방하여 에너지를 절약합니다.
*   **O(1) 상태 메모리 (State Memory)**: `ChunkedParallelRecurrence` 및 `FastSurprisal` 모듈을 사용하여 시퀀스 처리 중 메모리 점유율을 상수로 유지, 기존 어텐션의 2차 시간 복잡도를 회피합니다.
*   **회전식 위치 임베딩 (RoPE)**: v5.7에서 하이브리드 어텐션 블록에 RoPE를 이식하여 모델에 절대/상대 좌표 감각을 부여, 무한 반복 루프를 영구적으로 끊어냈습니다.

---

## 4. 실험 결과 및 디버깅 여정

### A. "외계어(Token Salad)" 현상 (표상 붕괴)
*   **문제**: 4,000 스텝에서 350M 모델이 의미 없는 외계어(`-alskman`, `omingiest`)를 출력하기 시작했습니다.
*   **진단 (탱크 불일치)**: 가혹한 대사 패널티로 인해 모델은 훈련 중 극도로 고갈된 탱크 상태($\sim 0.0135$)에서 생존하는 법을 배웠습니다. 그러나 추론(테스트) 시 탱크가 $1.0$(100%)으로 꽉 찬 상태로 시작되어 "급성 발작(Activation Explosion)"이 발생, 소프트맥스 확률이 파괴되었습니다.
*   **해결**: 추론 시 초기 탱크 용량을 안정적인 훈련 상태(`init_tank=0.0135`)로 하드코딩하여 유창한 영어 생성 능력을 즉시 복구했습니다.

### B. 기울기 누적(Gradient Accumulation)을 통한 OOM 해결
*   **문제**: 350M 모델을 24층으로 깊게 쌓자 40GB A100에서 메모리 초과(OOM) 에러가 발생했습니다.
*   **해결**: 배치 크기를 32에서 8로 줄이고 `grad_accum`을 32로 늘렸습니다. 실질적인 학습량은 동일하게 유지하면서 VRAM 점유율을 $\sim 25GB$ 수준으로 안정화하는 데 성공했습니다.

---

## 5. 향후 방향: Amygdala vs BitNet b1.58
마이크로소프트의 **BitNet b1.58**이 가중치를 3가지 숫자(-1, 0, 1)로 양자화하여 '곱셈을 없애는 방식'으로 효율성을 달성했다면, **Amygdala v5.7**은 놀람 수치에 따라 불필요한 '연산을 생략(Gating)하는 방식'으로 효율성을 달성합니다.

이 프로젝트의 궁극적인 목표는 이 둘을 결합하는 것입니다. 수학적 효율성을 위한 Ternary(3진) 가중치와 생물학적 효율성을 위한 대사 게이팅을 융합하여, 압도적으로 적은 전력을 소비하면서도 뛰어난 인지 능력을 갖춘 궁극의 모델을 만들 것입니다.
