# 🧠 AI Learning & Experimentation Space

This repository documents my **learning path, experiments, and hands-on experience** in Artificial Intelligence and Machine Learning.

The work here is primarily driven by studying well-known books and actively experimenting with their ideas through code. Rather than treating the material as static examples, I use it as a foundation for exploration, modification, and deeper understanding.

Each top-level folder in this repository is named after a book and contains chapter-based implementations and experiments.

---

## 📁 Repository Structure

- **Each folder name corresponds to a book**
- Inside each book folder:
  - Chapters are organized as `ch01`, `ch02`, `ch03`, ...
  - Each chapter typically contains:
    - A Python source file (`.py`)
    - A Jupyter Notebook (`.ipynb`)

📌 For readers and experimenters, **Jupyter Notebook files are recommended**.  
The original development, however, was mostly done using Python files inside a code editor.

---

## 📚 Books Covered

### *Machine Learning with Scikit-Learn, Keras & PyTorch*  
**Authors:** Sebastian Raschka, Yuxi (Hayden) Liu, Vahid Mirjalili

This book provides a practical and structured introduction to machine learning, progressing from fundamental concepts to modern deep learning workflows.

**High-level chapter overview:**
- Introduction to machine learning concepts, data preprocessing, and evaluation techniques
- Classical machine learning algorithms such as linear models, decision trees, ensemble methods, and dimensionality reduction
- Neural networks and deep learning using PyTorch, including training loops and optimization
- Strong emphasis on practical implementation alongside theory

**Repository layout for this book:**
- Chapters are stored as:
  - `ch01`, `ch02`, `ch03`, ...
- Each chapter folder contains:
  - A `.py` file with the main implementation
  - A corresponding `.ipynb` notebook for interactive use

---

## 🛠 Development Workflow

Although Jupyter Notebooks are included for accessibility, I personally prefer writing code in **Neovim**.

To maintain an interactive workflow while coding in `.py` files:
- I used **pywork.nvim** to run Python cells dynamically inside Neovim  
  🔗 https://github.com/jeryldev/pyworks.nvim
- I used **jupytext** to convert and synchronize `.py` files with Jupyter Notebooks

This setup enables:
- Cleaner, version-control-friendly Python code
- Notebook compatibility across different platforms
- A flexible workflow for both script-based and notebook-based users

---

## 📝 Notes

This repository is a **living workspace**.  
Code may include experiments, deviations from the book, or exploratory implementations, reflecting learning-in-progress rather than production-ready libraries.

Feel free to explore, experiment, and adapt the material 🚀

