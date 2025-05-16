# pd2pl Project Onboarding Guide

Welcome to the Pandas-to-Polars (pd2pl) translator project! This guide is designed to help you get up to speed quickly and start contributing.

## 1. Understand the Big Picture

Before diving into the code, it's important to understand what this project is about and how it works at a high level.

*   **Start here**: [Project Overview](./PROJECT_OVERVIEW.md)

## 2. Set Up Your Development Environment

Get your local environment configured so you can run the code, tests, and start experimenting.

*   **Setup instructions**: [Development Setup Guide](./DEVELOPMENT_SETUP.md)

## 3. Explore the Codebase

Familiarize yourself with how the project is organized, where key functionalities reside, and the conventions we follow.

*   **Code organization**: [Code Structure Guide](./CODE_STRUCTURE.md)

## 4. Learn the Core Concepts

Understand the fundamental principles behind the translation process, such as Abstract Syntax Trees (ASTs) and mapping strategies.

*   **Key principles**: [Core Concepts Guide](./CORE_CONCEPTS.md)

## 5. How to Contribute

Learn about our development workflow, how to pick up tasks, implement changes, and contribute your work back to the project.

*   **Contribution process**: [Contribution Workflow Guide](./CONTRIBUTION_WORKFLOW.md)

## 6. Deep Dive: Specific Architectures

For more complex parts of the system, we have dedicated architecture guides:

*   **Window Function Translation**: [Window Function Translation Guide](./WINDOW_FUNCTION_TRANSLATION.md)
    *   *(This covers the architecture for translating pandas window functions like `rolling()`, `expanding()`, and `ewm()` to their Polars equivalents.)*
*   **String Operations Translation**: [String Operations Translation Guide](./STRING_OPERATIONS.md)
    *   *(This covers the architecture for translating pandas string operations accessed through the `.str` accessor to their Polars equivalents.)*

## 7. Project Documentation

Existing design documents and task lists:

*   [TASKS.md](../../TASKS.md): Current list of features, bugs, and tasks.
*   [DESIGN.md](../../DESIGN.md): Original high-level design document.
*   [IMPORT_STRATEGIES.md](../../IMPORT_STRATEGIES.md): Strategies for handling Python imports.

## 8. Getting Help

If you have questions or get stuck:
*   Review the existing documentation and code.
*   Ask questions in the project's communication channels (e.g., issue tracker, Slack/Discord if available).
*   When asking, provide context: what you're trying to do, what you've tried, and any errors you're encountering.

We're excited to have you on board! 