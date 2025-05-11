import os

def visualize_barchart(model, top_n=10, output_dir="outputs") -> str:
    """
    Строит столбчатую диаграмму top_n тем и сохраняет HTML.
    Возвращает путь до файла.
    """
    fig = model.visualize_barchart(top_n_topics=top_n)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"top_{top_n}_topics.html")
    fig.write_html(path)
    return path

def visualize_hierarchy(model, output_dir="outputs") -> str:
    """
    Строит и сохраняет иерархическое дерево тем в HTML.
    """
    fig = model.visualize_hierarchy()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "topic_hierarchy.html")
    fig.write_html(path)
    return path
