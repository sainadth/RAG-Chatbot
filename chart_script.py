import plotly.graph_objects as go
import pandas as pd

# Load the data
data = [
  {"stage": "Input", "component": "Document Upload", "description": "PDF, DOCX, TXT files", "x": 1, "y": 8, "color": "blue"},
  {"stage": "Processing", "component": "Text Extraction", "description": "PyPDF2, python-docx", "x": 2.5, "y": 8, "color": "green"},
  {"stage": "Processing", "component": "Text Chunking", "description": "LangChain splitter", "x": 4, "y": 8, "color": "green"},
  {"stage": "Embedding", "component": "Embedding Generation", "description": "sentence-transformers", "x": 5.5, "y": 8, "color": "orange"},
  {"stage": "Storage", "component": "FAISS Vector Store", "description": "Vector database", "x": 7, "y": 8, "color": "purple"},
  {"stage": "Query", "component": "User Query", "description": "Natural language", "x": 1, "y": 4, "color": "blue"},
  {"stage": "Query", "component": "Query Embedding", "description": "Same embedding model", "x": 2.5, "y": 4, "color": "orange"},
  {"stage": "Retrieval", "component": "Similarity Search", "description": "FAISS cosine similarity", "x": 4, "y": 4, "color": "purple"},
  {"stage": "Retrieval", "component": "Context Retrieval", "description": "Top-k relevant chunks", "x": 5.5, "y": 4, "color": "purple"},
  {"stage": "Generation", "component": "LLM Generation", "description": "DialoGPT model", "x": 7, "y": 4, "color": "red"},
  {"stage": "Output", "component": "Response + Sources", "description": "Answer with citations", "x": 8.5, "y": 4, "color": "gold"}
]

df = pd.DataFrame(data)

# Map colors to exact brand colors
color_map = {
    "blue": "#1FB8CD",
    "green": "#2E8B57", 
    "orange": "#D2BA4C",
    "purple": "#5D878F",
    "red": "#DB4545",
    "gold": "#D2BA4C"
}

# Very short component names to fit in nodes (under 10 chars)
df['short_name'] = df['component'].map({
    'Document Upload': 'Upload',
    'Text Extraction': 'Extract',
    'Text Chunking': 'Chunk',
    'Embedding Generation': 'Embed',
    'FAISS Vector Store': 'Store',
    'User Query': 'Query',
    'Query Embedding': 'Q-Embed',
    'Similarity Search': 'Search',
    'Context Retrieval': 'Retrieve',
    'LLM Generation': 'Generate',
    'Response + Sources': 'Response'
})

# Create color mapping for stages
stage_colors = {
    'Input': '#1FB8CD',
    'Processing': '#2E8B57',
    'Embedding': '#D2BA4C', 
    'Storage': '#5D878F',
    'Query': '#1FB8CD',
    'Retrieval': '#5D878F',
    'Generation': '#DB4545',
    'Output': '#D2BA4C'
}

# Create the figure
fig = go.Figure()

# Add connecting lines for the document processing flow (top)
doc_flow_x = [1, 2.5, 4, 5.5, 7]
doc_flow_y = [8, 8, 8, 8, 8]
fig.add_trace(go.Scatter(
    x=doc_flow_x, y=doc_flow_y,
    mode='lines',
    line=dict(color='#333333', width=6),
    showlegend=False,
    hoverinfo='skip'
))

# Add connecting lines for the query flow (bottom)  
query_flow_x = [1, 2.5, 4, 5.5, 7, 8.5]
query_flow_y = [4, 4, 4, 4, 4, 4]
fig.add_trace(go.Scatter(
    x=query_flow_x, y=query_flow_y,
    mode='lines',
    line=dict(color='#333333', width=6),
    showlegend=False,  
    hoverinfo='skip'
))

# Add connecting line between FAISS Store and Similarity Search
fig.add_trace(go.Scatter(
    x=[7, 4], y=[8, 4],
    mode='lines',
    line=dict(color='#333333', width=6, dash='dot'),
    showlegend=False,
    hoverinfo='skip'
))

# Add components as scatter points grouped by stage
for stage in df['stage'].unique():
    stage_data = df[df['stage'] == stage]
    fig.add_trace(go.Scatter(
        x=stage_data['x'],
        y=stage_data['y'],
        mode='markers+text',
        marker=dict(
            size=50,
            color=stage_colors[stage],
            line=dict(width=4, color='white')
        ),
        text=stage_data['short_name'],
        textposition='middle center',
        textfont=dict(size=14, color='white', family='Arial Black'),
        name=stage,
        hovertemplate='<b>%{text}</b><br>Stage: ' + stage + '<br>Full: ' + stage_data['component'].iloc[0] + '<extra></extra>',
        cliponaxis=False
    ))

# Update layout with better spacing
fig.update_layout(
    title='RAG Pipeline Architecture',
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[0, 9.5]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[2.5, 9.5]
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
        font=dict(size=12)
    ),
    plot_bgcolor='white'
)

# Save the chart
fig.write_image('rag_pipeline_flowchart.png')
print("Chart saved as rag_pipeline_flowchart.png")