# CausalMind (Fighter-CFLM v2.8)
**Evolution from Statistical NLP to Causal Flight Dynamics AI**

[Read in Korean (한국어 번역)](#-한국어-번역-korean-translation)

## 1. Project Overview & Motivation
The project began as a 350M-parameter language model aiming to achieve cognitive intelligence. However, during 4,000 steps of statistical memorization training ($L_{ce}$), it suffered from "Semantic Collapse" (e.g., generating gibberish like `-alskman...`) and severe logical hallucinations. It became clear that simply increasing parameters and relying on statistical frequency (Cross-Entropy) does not lead to true causal understanding.

To overcome this, the project underwent a radical paradigm shift:
1.  **Architecture**: Shifted from discrete Transformer/RNN layers to a **Continuous Fractal Language Model (CFLM)**.
2.  **Training**: Abandoned 50,000-epoch backpropagation in favor of **Zero-Epoch Few-Shot Learning (FCLAP)** and **Active Free Energy Tournament Loss (AFETL)** during a simulated "Sleep Phase".
3.  **Domain**: Pivoted from text prediction to real-time flight dynamics (**2026 AI Pilot Top Gun Challenge**), shrinking the model from 350M to 15M parameters to enable $\mathcal{O}(1)$ real-time closed-loop control.

---

## 2. Theoretical Foundations & Mathematical Formulas

### A. Dual Memory System & FastSurprisal
The architecture separates memory into two distinct weights:
*   $\theta_{fixed}$ (**Slow Weights**): Frozen core knowledge representing fundamental physical laws and universal structures. Prevents catastrophic forgetting.
*   $\theta_{plastic}$ (**Fast Weights**): A dynamic cache that instantly records new causal relationships based on "Cognitive Surprisal" (Prediction Error).

### B. Continuous Fractal Language Model (CFLM)
Instead of passing through discrete layers, the input state $z_0$ integrates over time $t \in [0, 1]$ using an Ordinary Differential Equation (ODE). It features a metabolic penalty to ensure real-time latency.

$$z_{out} = z_0 + \int_{0}^{1} \left( f_{fractal}(z, t; \theta_{fixed}) + \alpha \cdot \text{ReLU}(\mathcal{S}_t - \tau) \cdot g(z, t; \theta_{plastic}) \right) dt$$

*   $\alpha \cdot \text{ReLU}(\mathcal{S}_t - \tau)$: Endocrine gating mechanism that activates the plastic weights only when the surprisal $\mathcal{S}_t$ exceeds threshold $\tau$.

### C. Active Free Energy Tournament Loss (AFETL) - Multiverse Sleep
During the "Night Phase", the model consolidates $\theta_{plastic}$ into $\theta_{fixed}$ by simulating $K$ mutant agents ($\Delta \theta_k$) in a virtual sandbox. The ultimate referee is the AFETL equation:

$$\mathcal{L}(\Delta \theta_k) = \underbrace{-\log P(x_{plastic} \mid \theta_{fixed} + \Delta \theta_k)}_{\text{(1) Progressiveness}} + \underbrace{\lambda_1 \|\Delta \theta_k\|_1}_{\text{(2) Economy}} + \underbrace{\lambda_2 \mathcal{D}_{KL}\left( P_{\theta_{fixed} + \Delta \theta_k} \| P_{\theta_{fixed}} \right)_{\text{on } \mathcal{M}_{safe}}}_{\text{(3) Conservativeness (Immunity)}}$$

Only the agent with the lowest energy (lowest AFETL score) is merged into the long-term core memory.

---

## 3. Domain Pivot: AI Pilot Challenge

To participate in the combat flight simulation challenge, the I/O interface was completely overhauled to handle physical vectors instead of language tokens.

| Component | Legacy SLLM (350M) | Fighter-CFLM v2.8 (15M) |
| :--- | :--- | :--- |
| **Input (Cause)** | 50,000 Vocab Token IDs | 12D State Vector (Coord, Vel, Accel, G-load, Relative Target) |
| **Output (Effect)**| Next Token Probabilities | 4D Control Vector (Pitch, Roll, Yaw, Throttle) |
| **Inference Time**| $\mathcal{O}(N^2)$ (Heavy) | $\mathcal{O}(1)$ (Real-time, <1ms on CPU) |
| **Environment** | Text Corpora | JSBSim Flight Dynamics Engine |

---

## 4. System Implementation & File Structure

The project operates on a "Day (Real-time Simulation) / Night (Cloud Consolidation)" cycle:

1.  **`neuro_flight_env.py`**: The JSBSim environment wrapper. Extracts the 12D state vector and normalizes it for the CFLM.
2.  **`cflm_core.py`**: The core continuous fractal language model architecture.
3.  **`afetl_sleep_trainer.py`**: The night-phase orchestrator. Spawns multiverse models, and applies AFETL to evolve weights.
4.  **`generate_graphs.py`**: Visualizes the evolution of flight stability across phases.

---

## 5. Experimental Results: The Evolution of Flight

![Flight Evolution](flight_evolution_graph.png)

### Phase 1: Motor Babbling (Day 1)
*   **Status**: Initialized with random/basic physics weights.
*   **Result**: The AI exhibited "motor babbling", shaking the controls and bouncing off the ground at 7.8ft (Step 80) before touching down erratically. Generated high surprisal data.

### Phase 2: First AFETL Consolidation (Night 1)
*   **Environment**: Google Colab T4.
*   **Result**: Processed the bouncing logs. The AFETL successfully found a topological path linking control inputs to altitude maintenance. Loss dropped significantly to `-0.0199`. Generated `evolved_weights.pt`.

### Phase 3: High-Speed Stable Takeoff Roll (Day 2)
*   **Result**: Complete behavioral shift. The model learned "Energy Conservation". It held stable ground effect (5.2ft ~ 5.4ft) and accelerated smoothly up to **194.0 kts (Step 900)** without bouncing. 
*   **Next Milestone**: Apply an altitude threshold reward in AFETL to teach the model to pull the elevator (Rotate) at 190kts for full takeoff.

<br>
<br>

---

# 🇰🇷 한국어 번역 (Korean Translation)

# CausalMind (Fighter-CFLM v2.8)
**통계적 NLP에서 인과적 비행 동역학 AI로의 진화**

## 1. 프로젝트 개요 및 동기
이 프로젝트는 초기에 인지적 지능을 달성하기 위한 350M 파라미터의 언어 모델로 시작되었습니다. 그러나 통계적 암기 학습($L_{ce}$)을 4,000 스텝 진행하는 동안 "의미론적 붕괴(Semantic Collapse)"(예: `-alskman...` 같은 외계어 생성)와 심각한 논리적 환각(Hallucination) 현상을 겪었습니다. 단순히 파라미터를 늘리고 교차 엔트로피(Cross-Entropy) 같은 통계적 빈도에 의존하는 것으로는 진정한 인과적 이해에 도달할 수 없다는 것이 명백해졌습니다.

이를 극복하기 위해 프로젝트는 급진적인 패러다임 전환을 겪었습니다:
1.  **아키텍처**: 이산적인 트랜스포머/RNN 레이어에서 **연속 프랙탈 언어 모델(CFLM, Continuous Fractal Language Model)**로 전환.
2.  **훈련 방식**: 50,000 에폭의 역전파를 버리고, 시뮬레이션된 "수면 단계(Sleep Phase)"에서 **Zero-Epoch Few-Shot Learning (FCLAP)** 및 **능동적 자유 에너지 토너먼트 손실(AFETL)**을 도입.
3.  **도메인**: 텍스트 예측에서 실시간 비행 동역학(**2026 AI Pilot Top Gun Challenge**)으로 전환. 실시간 폐루프(Closed-loop) 제어를 위해 파라미터를 350M에서 15M으로 경량화($\mathcal{O}(1)$ 연산).

---

## 2. 이론적 기반 및 수학적 공식

### A. 이중 메모리 시스템 및 빠른 놀람(FastSurprisal)
이 아키텍처는 메모리를 두 가지 가중치로 분리합니다:
*   $\theta_{fixed}$ (**느린 가중치**): 근본적인 물리 법칙과 보편적 구조를 나타내는 고정된 핵심 지식. 파국적 망각(Catastrophic forgetting)을 방지합니다.
*   $\theta_{plastic}$ (**빠른 가중치**): "인지적 놀람(Prediction Error)"에 기반하여 새로운 인과 관계를 즉각적으로 기록하는 동적 캐시.

### B. 연속 프랙탈 언어 모델 (CFLM)
이산적인 레이어를 통과하는 대신, 입력 상태 $z_0$는 상미분 방정식(ODE)을 사용하여 시간 $t \in [0, 1]$에 대해 적분됩니다. 실시간 레이턴시를 보장하기 위한 대사 패널티(Metabolic penalty)가 특징입니다.

$$z_{out} = z_0 + \int_{0}^{1} \left( f_{fractal}(z, t; \theta_{fixed}) + \alpha \cdot \text{ReLU}(\mathcal{S}_t - \tau) \cdot g(z, t; \theta_{plastic}) \right) dt$$

*   $\alpha \cdot \text{ReLU}(\mathcal{S}_t - \tau)$: 놀람 $\mathcal{S}_t$가 임계값 $\tau$를 초과할 때만 가소성 가중치(Plastic weights)를 활성화하는 내분비 게이팅 메커니즘.

### C. 능동적 자유 에너지 토너먼트 손실 (AFETL) - 다중 우주 수면
"밤(Night Phase)" 동안 모델은 가상 샌드박스에서 $K$개의 돌연변이 에이전트($\Delta \theta_k$)를 시뮬레이션하여 $\theta_{plastic}$을 $\theta_{fixed}$로 통합합니다. 최종 심판은 AFETL 방정식입니다:

$$\mathcal{L}(\Delta \theta_k) = \underbrace{-\log P(x_{plastic} \mid \theta_{fixed} + \Delta \theta_k)}_{\text{(1) 진보성}} + \underbrace{\lambda_1 \|\Delta \theta_k\|_1}_{\text{(2) 경제성}} + \underbrace{\lambda_2 \mathcal{D}_{KL}\left( P_{\theta_{fixed} + \Delta \theta_k} \| P_{\theta_{fixed}} \right)_{\text{on } \mathcal{M}_{safe}}}_{\text{(3) 보수성 (면역력)}}$$

가장 낮은 에너지(가장 낮은 AFETL 점수)를 가진 에이전트만이 장기 핵심 기억에 병합됩니다.

---

## 3. 도메인 전환: AI 파일럿 챌린지

전투 비행 시뮬레이션 챌린지에 참여하기 위해, 언어 토큰 대신 물리 벡터를 처리하도록 I/O 인터페이스를 완전히 개조했습니다.

| 컴포넌트 | 레거시 SLLM (350M) | Fighter-CFLM v2.8 (15M) |
| :--- | :--- | :--- |
| **입력 (원인)** | 50,000 단어 토큰 ID | 12D 상태 벡터 (좌표, 속도, 가속도, G-포스, 상대 타겟) |
| **출력 (결과)** | 다음 토큰 확률 | 4D 제어 벡터 (피치, 롤, 요, 스로틀) |
| **추론 속도** | $\mathcal{O}(N^2)$ (무거움) | $\mathcal{O}(1)$ (실시간, CPU에서 1ms 미만) |
| **환경** | 텍스트 코퍼스 | JSBSim 비행 동역학 엔진 |

---

## 4. 시스템 구현 및 파일 구조

이 프로젝트는 "낮(실시간 시뮬레이션) / 밤(클라우드 통합)" 사이클로 작동합니다:

1.  **`neuro_flight_env.py`**: JSBSim 환경 래퍼. 12D 상태 벡터를 추출하고 CFLM에 맞게 정규화합니다.
2.  **`cflm_core.py`**: 핵심 연속 프랙탈 언어 모델 아키텍처 모듈.
3.  **`afetl_sleep_trainer.py`**: 밤 단계(Night-phase) 오케스트레이터. 다중 우주 모델을 생성하고 AFETL을 적용하여 가중치를 진화시킵니다.
4.  **`generate_graphs.py`**: 비행 안정성의 진화 과정을 시각화하는 그래프 스크립트.

---

## 5. 실험 결과: 비행의 진화

![Flight Evolution](flight_evolution_graph.png)

### 1단계: 운동 옹알이 (Motor Babbling) - 1일 차
*   **상태**: 무작위/기본 물리 가중치로 초기화.
*   **결과**: AI는 조종간을 흔들며 지면에서 7.8ft 높이로 튕겨 오르는(Step 80) 등 불규칙하게 착륙하는 "운동 옹알이"를 보였습니다. 높은 놀람(Surprisal) 데이터를 생성했습니다.

### 2단계: 첫 번째 AFETL 통합 (Night 1)
*   **환경**: Google Colab T4.
*   **결과**: 튕겨 오르는 로그를 처리했습니다. AFETL은 제어 입력과 고도 유지를 연결하는 위상적 경로를 성공적으로 찾았습니다. 손실이 `-0.0199`로 크게 떨어졌습니다. `evolved_weights.pt` 생성.

### 3단계: 고속 안정적 이륙 활주 (Day 2)
*   **결과**: 완전한 행동의 변화. 모델이 "에너지 보존"을 학습했습니다. 안정적인 지면 효과(5.2ft ~ 5.4ft)를 유지하며 통통 튀지 않고 **194.0 kts (Step 900)**까지 부드럽게 가속했습니다.
*   **다음 목표**: AFETL에 고도 임계값 보상을 적용하여 190kts에서 엘리베이터(기수 올림)를 당겨 완전한 이륙을 하도록 모델을 학습시킵니다.
