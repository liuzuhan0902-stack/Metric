# MetricDB MSc Cluster Project (2025/26)

This repository provides a foundational framework and code template for the MetricDB MSc cluster project. MetricDB is a research-oriented database system designed to decouple "logical similarity" from "physical embedding," enabling more flexible and robust vector search.

---

## 🚀 Getting Started

### 1. Environment Setup (Recommended: uv)
Virtual environments are essential for Python development. They create isolated spaces for your project's dependencies, preventing conflicts between different projects and your system-wide Python installation.

We recommend using [**uv**](https://github.com/astral-sh/uv), an extremely fast Python package and gRPC-based virtual environment manager.

**Setting up with uv:**
1. **Install uv** (if you haven't): Follow the [official installation guide](https://github.com/astral-sh/uv#installation).
2. **Create and activate a virtual environment:**
   ```bash
   uv venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

**Alternative (Standard pip):**
If you prefer standard tools, ensure you have a virtual environment active, then run:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
The system relies on synthetic datasets for development and testing. You can generate data with custom scales:
```bash
# Default: 100 samples, 128 dimensions
python data/data_generator.py
```
To change the scale, modify `DATA_SIZE` and `VECTOR_DIM` in `main.py` and run it.

### 3. Run the Integrated Demo
The `main.py` script demonstrates how all five modules connect. It serves as a starting point to see how your module interacts with others.
```bash
python main.py
```

### 4. Basic Interface Example
If you want to see a simplified, standalone example of how to implement and use the `MetricRelation` interface, check the `examples/` directory:
```bash
python examples/basic_usage.py
```

### 5. Performance Optimization (Optional: C++)
For performance-critical tasks (e.g., custom distance calculations or index traversal), you may want to implement core logic in C++. We provide a simple example of Python-C++ interoperability using `ctypes`:
- **Example Directory:** `examples/cpp_extension/`
- **Usage:** Follow the compilation instructions in `fast_distance.cpp` (recommend adding `-static` on Windows to avoid DLL dependency issues), then run `test_cpp_ext.py`.

### 6. Run Tests
Ensure your implementation doesn't break existing interfaces.
```bash
# Run all tests
python -m unittest discover tests

# Run a specific module's test (e.g., Module 1)
python -m unittest tests/test_predicate_search.py

# Run the integration tests
python -m unittest tests/test_integration.py
```

---

## 🛠 Project Structure & Module Tasks

### `core/` - The Logical Skeleton
Contains `base.py` which defines the interfaces (`MetricRelation`, `BaseMetricIndex`, etc.) that all students must adhere to. **Do not modify these interfaces without consulting your supervisor.**

---

## 🧪 Development & Testing Workflow

This project is designed for **concurrent development**. Students can work on their respective modules independently without waiting for others.

### Phase 1: Independent Development (Unit Testing)
Each module has its own test suite in `tests/test_<module_name>.py`. These tests are designed to be self-contained:

1.  **Isolation via Mocks**: In `tests/test_predicate_search.py`, for example, we use a `MockIndex` class. This allows the student working on **Module 1** to test their filtering logic even if **Module 4's** index implementation is not yet complete.
2.  **How to test your module**:
    ```bash
    # Replace <module_name> with your assigned module (e.g., predicate_search)
    python -m unittest tests/test_predicate_search.py
    ```
3.  **Task**: As you implement your `TODOs`, update the corresponding test file to reflect your logic and ensure your code is robust.

### Phase 2: Collaborative Integration (Integration Testing)
Once you are confident in your module's functionality, you can test how it interacts with the rest of the system:

1.  **Shared Components**: All modules are instantiated and connected in `main.py`. Running `python main.py` is the easiest way to see the system in action.
2.  **Running Integration Tests**:
    ```bash
    python -m unittest tests/test_integration.py
    ```
    The integration tests simulate end-to-end scenarios (e.g., SQL -> Index -> Search).
3.  **Cross-Module Testing**: If you want to test your module (e.g., Module 1) using a teammate's real implementation (e.g., Module 4) instead of a Mock, simply modify your unit test to import their class instead of using the Mock version.

---

## 📂 Student Assignments (Modules)

To help students quickly locate their tasks, the following table summarizes the assignments and the corresponding code files:

| Module | Task Name | Student Role | Primary File Location |
| :--- | :--- | :--- | :--- |
| **Module 1** | Predicate Vector Search | Hybrid Search Developer | `modules/predicate_search/engine.py` |
| **Module 2** | Metric SQL Engine | Query Engine Developer | `modules/query_engine/parser.py` |
| **Module 3** | Virtual & Hard Unions | Multi-DB Manager | `modules/union_management/union.py` |
| **Module 4** | Metric Indexing | Indexing Specialist | `modules/metric_indexing/indexing.py` |
| **Module 5** | Geometric Hashing | Entity Linking Specialist | `modules/entity_linking/linker.py` |

---

### Module Details

#### 1. `predicate_search/` (Module 1: Predicate Vector Search)
- **Assigned to:** Student 1
- **Objective:** Implement hybrid search that combines vector similarity with relational predicates (e.g., SQL `WHERE` clauses).
- **Key Task:** Develop efficient pre-filtering or post-filtering logic to narrow down results.
- **Entry Point:** `modules/predicate_search/engine.py`

#### 2. `query_engine/` (Module 2: Metric SQL Engine)
- **Assigned to:** Student 2
- **Objective:** Extend SQL syntax to support metric operators (like `SIMILAR TO`) and optimize logical query plans.
- **Key Task:** Implement a parser that handles metric similarity and an optimizer that decides the best execution strategy.
- **Entry Point:** `modules/query_engine/parser.py`

#### 3. `union_management/` (Module 3: Virtual & Hard Unions)
- **Assigned to:** Student 3
- **Objective:** Handle queries and data integration across multiple MetricDB instances or embedding spaces.
- **Key Task:** Implement "Soft Union" (query translation) and "Hard Union" (physical data alignment).
- **Entry Point:** `modules/union_management/union.py`

#### 4. `metric_indexing/` (Module 4: Metric Indexing)
- **Assigned to:** Student 4
- **Objective:** Build index structures based on metric properties (distances) rather than fixed coordinates.
- **Key Task:** Implement distance-based structures (e.g., M-Tree, VP-Tree) or wrappers for coordinate-based indices (e.g., FAISS).
- **Entry Point:** `modules/metric_indexing/indexing.py`

#### 5. `entity_linking/` (Module 5: Geometric Hashing)
- **Assigned to:** Student 5
- **Objective:** Discover links between entities in different vector spaces without shared IDs.
- **Key Task:** Implement geometric hashing to find "anchor points" and link related items based on their metric relative positions.
- **Entry Point:** `modules/entity_linking/linker.py`

---

## 📝 Contribution Guidelines

1. **Module Independence:** Your code should reside primarily within your assigned `modules/` folder.
2. **Follow Interfaces:** Use the abstract base classes in `core/base.py`.
3. **Write Tests:** Add meaningful unit tests in `tests/test_<your_module>.py`.
4. **Documentation:** Use English docstrings for all classes and methods. Keep comments detailed to help your teammates understand your logic.

---

## 📊 Data & Testing
- Use `data/data_generator.py` to create datasets for your experiments.
- The `tests/` directory contains templates for unit testing. **You should expand these as you develop your features.**

---

## 📂 Repository Standards

To maintain a clean and professional development environment, this repository includes several standard files:

- **`requirements.txt`**: Lists all Python libraries required to run the project. Use `uv pip install -r requirements.txt` (or `pip install`) to sync your environment.
- **`.gitignore`**: Tells Git which files/directories to ignore (e.g., `__pycache__`, large generated datasets in `data/`, IDE settings). This prevents "junk" files from being committed to the repository.
- **`LICENSE`**: Specifies the terms under which the code can be used and distributed (MIT License).

---

## 🤝 Collaboration & Git Basics

Since multiple students are working on independent modules within the same repository, following a basic Git workflow is essential:

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   ```
2. **Create a Branch for Your Module:**
   Do not work directly on the `main` branch. Create a feature branch instead:
   ```bash
   git checkout -b feature/module-1-predicate-search
   ```
3. **Commit Your Changes:**
   Keep your commits focused and provide clear messages:
   ```bash
   git add modules/predicate_search/
   git commit -m "Implement initial pre-filtering logic for Module 1"
   ```
4. **Stay Updated:**
   Frequently pull changes from the main repository to avoid complex merge conflicts:
   ```bash
   git pull origin main
   ```

**External Resources for Learning Git:**
- [Git Handbook (GitHub)](https://guides.github.com/introduction/git-handbook/)
- [Visualizing Git Concepts](https://learngitbranching.js.org/)
- [Git Official Documentation](https://git-scm.com/doc)
