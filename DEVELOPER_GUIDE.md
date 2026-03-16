# Portfolio Developer Guide

This guide explains how to add new projects to your portfolio manually. 

## Table of Contents
1. [Adding a Live Demo (External Link)](#1-adding-a-live-demo-external-link)
2. [Adding a GitHub-Only Project](#2-adding-a-github-only-project)
3. [Adding a Live ML Algorithm (Internal Pipeline)](#3-adding-a-live-ml-algorithm-internal-pipeline)

---

## 1. Adding a Live Demo (External Link)
Use this for projects hosted on Render, Netlify, or Vercel.

**Steps:**
1. Open `templates/index.html`.
2. Locate the `<div class="projects-grid">` section.
3. Copy an existing project card (the `<a>` tag with class `project-card`).
4. Update the following:
   - `href`: The URL of your live project.
   - `.project-title`: The name of the project.
   - `.project-desc`: A 1-2 sentence description.
   - `.tech-chip`s: Update the spans with technologies like `Python`, `React`, etc.

---

## 2. Adding a GitHub-Only Project
Use this to showcase code for projects that don't have a live website.

**Steps:**
1. Follow the same steps as the "Live Demo" above.
2. Direct the `href` to your GitHub repository.
3. Update the button text inside the card:
   - Find `<span class="plink plink-demo">Live Demo &#8594;</span>`.
   - Change it to `<span class="plink plink-demo">View Code &#8594;</span>`.

---

## 3. Adding a Live ML Algorithm (Internal Pipeline)
Use this if you want a local form where users input data and see a prediction (like the "Student Performance" project).

### Step A: Model & Pipeline
1. Save your trained `.pkl` or model files in the `artifacts/` directory.
2. In `src/pipeline/predict_pipeline.py`, create two classes:
   - `NewProjectData`: To map form inputs to a DataFrame.
   - `NewProjectPipeline`: To load the model and call `.predict()`.

### Step B: Create the Form
1. Create a new HTML file in `templates/` (e.g., `heart_disease.html`).
2. Build a `<form>` with `method="POST"`. Ensure `name` attributes match your `NewProjectData` class.

### Step C: Update the Backend (`app.py`)
Add a new route to handle the page and the prediction:

```python
@app.route("/new-project", methods=['GET', 'POST'])
def new_project():
    if request.method == "GET":
        return render_template('your_form.html')
    else:
        from src.pipeline.predict_pipeline import NewProjectPipeline, NewProjectData
        data = NewProjectData(
            feature1 = request.form.get('feature1'),
            feature2 = request.form.get('feature2')
        )
        df = data.get_data_as_dataframe()
        predict_pipeline = NewProjectPipeline()
        results = predict_pipeline.predict(df)
        return render_template('your_form.html', results=results)
```

### Step D: Link from Homepage
1. In `templates/index.html`, add a project card.
2. Set the `href` to your new route: `href="/new-project"`.

---

## Design Tips
- **Colors**: Use `top-blue` or `top-orange` classes on the project card to match the existing theme.
- **Icons**: You can find emojis or icons for the `.project-cat` span here: [emojipedia.org](https://emojipedia.org).
