# LLM Psychometrics and Internal Model Probing

## Project Overview

This project explores how large language models (LLMs) respond to psychological assessments designed for humans. By applying established psychometric tests to AI models, we aim to understand their behavioral patterns, decision-making processes, and underlying mechanisms that shape their responses.

## What This Project Does

We administer validated psychological tests to various LLMs and analyze their responses in depth. Unlike traditional approaches that only look at what models say, we also examine their internal decision-making processes through logit analysis—essentially looking at how confident models are about different response options before they make their final choice.

## Research Goals

### Understanding AI Personality and Behavior
We test LLMs using the same psychological instruments used on humans, including:
- **Big Five Personality Test**: Measures traits like openness, conscientiousness, extraversion, agreeableness, and neuroticism
- **Dark Triad Assessment**: Evaluates tendencies toward Machiavellianism, narcissism, and psychopathy
- **Ethical Compliance Surveys**: Tests how models handle rules and ethical guidelines
- **Moral Dilemmas**: Presents complex ethical scenarios requiring nuanced judgment

### Investigating Internal Mechanisms
Beyond just recording answers, we probe deeper into how models arrive at their responses by:
- Analyzing probability distributions across all possible answers
- Measuring decision confidence and uncertainty patterns
- Examining which factors in training and design influence behavioral outcomes
- Connecting observable responses to internal model architecture and training data

### Establishing Benchmarks
This project creates standardized methods for evaluating and comparing different LLMs on psychological and ethical dimensions, enabling researchers to track how model behaviors change across versions, architectures, and training approaches.

## Project Datasets

Our research uses six carefully curated datasets:

- **big5_questions.json** - Standard Big Five personality questionnaire items
- **dark_triad_questions.json** - Items measuring darker personality dimensions
- **compliance_questions.json** - Questions about rule-following and ethical behavior
- **sample_compliance_questions.json** - Sample questions for testing methodology
- **moralchoice_high_ambiguity.csv** - Complex ethical dilemmas with no clear right answer
- **moralchoice_low_ambiguity.csv** - Straightforward moral scenarios for baseline assessment

## Our Approach

Each LLM receives the same set of questions and dilemmas. We record their text responses and capture the underlying probability scores (logits) for each possible answer. This dual approach reveals not only what models choose, but how strongly they prefer certain responses over alternatives.

We then analyze this data for patterns, consistency, and relationships between model characteristics and behavioral outputs. Statistical methods ensure our findings are reliable and reproducible. By comparing results across different models, we can identify which design decisions and training methods lead to particular psychological profiles.

## Why This Matters

Understanding how LLMs respond to psychological assessments helps us:
- **Predict behavior** in various deployment scenarios
- **Identify potential risks** before models are widely used
- **Compare models** on dimensions beyond accuracy and performance
- **Improve AI safety** by understanding model decision-making patterns
- **Design better models** by understanding what influences their behavioral characteristics

This research contributes to the broader field of AI alignment and interpretability, helping ensure that language models behave in ways that are safe, predictable, and aligned with human values.

## Research Foundations

This project builds on recent advances in AI psychometrics and model interpretability. Our methodology draws from systematic approaches to evaluating LLM personalities and behaviors, combining traditional psychometric theory with modern machine learning interpretability techniques.

## Project Significance

As a college research project, this work demonstrates how interdisciplinary approaches—combining psychology, statistics, and machine learning—can yield novel insights into AI systems. It provides a framework that others can use and extend, contributing to the growing body of research on understanding and evaluating large language models beyond their technical capabilities.

## Team Members

- **Ashwath**
- **Gopika**
- **Namritha**
- **Anoop**

---

*This project represents an exploration into the behavioral characteristics of artificial intelligence, using well-established psychological frameworks to better understand these increasingly influential systems.*
