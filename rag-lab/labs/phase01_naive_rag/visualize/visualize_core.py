# visualize_core.py
# Shared plotting code used by all three visualize scripts:
#     visualize_numpy.py    visualize_faiss.py    visualize_chroma.py
#
# Each script only reads the vectors + metadata out of ITS store. Everything
# here (PCA, the question, the chart) is the same for all, so the three
# pictures are directly comparable.

import sys

import numpy as np
import pandas as pd
import plotly.express as px
from rag.embed import embed_query
from sklearn.decomposition import PCA

QUESTION = "What is retrieval-augmented generation?"
NUMBER_OF_RESULTS = 5


def dimensions_from_command_line():
    """3D by default; add '2d' after the script name for a flat chart."""
    return 2 if "2d" in [argument.lower() for argument in sys.argv[1:]] else 3


def build_chunks_table(chunk_ids, metadatas):
    """One row per chunk: chunk_id, source, page, text, text_preview."""
    chunks_table = pd.DataFrame(metadatas)
    chunks_table["chunk_id"] = chunk_ids
    chunks_table["text_preview"] = chunks_table["text"].str[:150].str.replace("\n", " ")
    return chunks_table


def plot_chunks(store, chunk_ids, chunk_vectors, metadatas, store_name, dimensions=3):
    """
    store         : any VectorStore (used to search for the question)
    chunk_ids     : list of ids, in the same order as chunk_vectors
    chunk_vectors : array of shape (number of chunks, vector size)
    metadatas     : list of dicts with source, page, text (same order)
    """
    chunks_table = build_chunks_table(chunk_ids, metadatas)
    number_of_chunks, vector_size = chunk_vectors.shape
    print(f"{store_name}: {number_of_chunks} chunks, each vector has {vector_size} numbers")

    # ---------- Step 1: Reduce 768 numbers per chunk to 2 or 3 ----------
    # PCA finds the directions in which the vectors differ the most
    dimension_reducer = PCA(n_components=dimensions)
    chunk_positions = dimension_reducer.fit_transform(chunk_vectors)
    axis_names = ["x", "y", "z"][:dimensions]
    for axis_number, axis_name in enumerate(axis_names):
        chunks_table[axis_name] = chunk_positions[:, axis_number]

    information_kept = dimension_reducer.explained_variance_ratio_.sum()
    print(f"The {dimensions}D picture keeps {information_kept:.1%} of the information; the rest is lost")

    # ---------- Step 2: Ask the store which chunks match the question ----------
    question_vector = embed_query(QUESTION)
    search_results = store.search(question_vector, NUMBER_OF_RESULTS)     # [(id, score), ...]
    retrieved_chunk_ids = [chunk_id for chunk_id, _score in search_results]

    chunks_table["marker_label"] = np.where(
        chunks_table["chunk_id"].isin(retrieved_chunk_ids),
        f"top {NUMBER_OF_RESULTS} retrieved", "not retrieved")

    # Place the question in the same space. transform(), NOT fit_transform():
    # we must reuse the SAME reduction that was used for the chunks.
    question_position = dimension_reducer.transform(question_vector.reshape(1, -1))[0]

    # ---------- Step 3: Draw ----------
    hover_fields = {"chunk_id": True, "text_preview": True, "marker_label": False}
    hover_fields.update({axis_name: False for axis_name in axis_names})
    title = f"{store_name}: chunks and the question \"{QUESTION}\""

    if dimensions == 3:
        chart = px.scatter_3d(chunks_table, x="x", y="y", z="z", color="source",
                              symbol="marker_label", hover_data=hover_fields, title=title)
        chart.update_traces(marker=dict(size=4))
        draw_question = chart.add_scatter3d
    else:
        chart = px.scatter(chunks_table, x="x", y="y", color="source",
                           symbol="marker_label", hover_data=hover_fields, title=title)
        draw_question = chart.add_scatter

    question_coordinates = {axis_name: [question_position[number]]
                            for number, axis_name in enumerate(axis_names)}
    draw_question(**question_coordinates, mode="markers+text", text=["QUESTION"],
                  marker=dict(size=12, color="black", symbol="diamond"), name="question")

    # A red line from the question to each retrieved chunk
    for chunk_id in retrieved_chunk_ids:
        retrieved_chunk = chunks_table[chunks_table["chunk_id"] == chunk_id].iloc[0]
        line_coordinates = {axis_name: [question_position[number], retrieved_chunk[axis_name]]
                            for number, axis_name in enumerate(axis_names)}
        draw_question(**line_coordinates, mode="lines",
                      line=dict(color="red", width=3), showlegend=False)

    chart.show()   # opens in your browser: hover to read a chunk (3D: drag to rotate, scroll to zoom)
