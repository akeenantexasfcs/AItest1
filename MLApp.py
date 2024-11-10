#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import json
import uuid
import pandas as pd
from typing import Dict, List, Union
import io

def process_textract_json(textract_data: Dict) -> List[Dict]:
    """Process Textract JSON into flattened format for OpenSearch"""
    if isinstance(textract_data, str):
        textract_data = json.loads(textract_data)
    
    documents = []
    doc_id = f"textract_doc_{uuid.uuid4().hex[:8]}"
    
    # Process each block from Textract
    for block in textract_data.get("Blocks", []):
        if block.get("BlockType") in ["LINE", "WORD"] and block.get("Text"):
            doc = {
                "DocumentId": doc_id,
                "BlockType": block.get("BlockType", "LINE"),
                "Id": block.get("Id", ""),
                "Text": block.get("Text", ""),
                "Confidence": block.get("Confidence", 99.0),
                "Page": block.get("Page", 1)
            }
            documents.append(doc)
    
    return documents

def create_bulk_index_entries(documents: List[Dict], index_name: str) -> str:
    """Create bulk index entries for OpenSearch"""
    bulk_entries = []
    
    for doc in documents:
        # Create index action
        action = {
            "index": {
                "_index": index_name
            }
        }
        
        # Add both action and document
        bulk_entries.append(json.dumps(action))
        bulk_entries.append(json.dumps(doc))
    
    return "\n".join(bulk_entries)

st.title("Textract JSON to OpenSearch Bulk Index Converter")

# File upload or JSON input
input_type = st.radio("Choose input type:", ["File Upload", "Paste JSON"])

textract_data = None

if input_type == "File Upload":
    uploaded_file = st.file_uploader("Upload Textract JSON file", type=['json'])
    if uploaded_file:
        try:
            textract_data = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")
else:
    json_input = st.text_area("Paste Textract JSON here:", height=300)
    if json_input:
        try:
            textract_data = json.loads(json_input)
        except Exception as e:
            st.error(f"Error parsing JSON: {str(e)}")

# Index name input
index_name = st.text_input("Enter Index Name:", value="textract-documents")

if st.button("Convert to Bulk Index Format"):
    if textract_data:
        try:
            # Process Textract JSON
            documents = process_textract_json(textract_data)
            
            # Create bulk index entries
            bulk_index_content = create_bulk_index_entries(documents, index_name)
            
            # Display stats
            st.info(f"Processed {len(documents)} text blocks from Textract")
            
            # Display the output
            st.text_area("Bulk Index Format (Copy this to OpenSearch):", 
                        bulk_index_content, 
                        height=300)
            
            # Download button
            st.download_button(
                label="Download Bulk Index File",
                data=bulk_index_content,
                file_name=f"textract_bulk_index.json",
                mime="application/json"
            )
            
            # Preview of processed data
            if st.checkbox("Show preview of processed data"):
                df = pd.DataFrame(documents)
                st.dataframe(df)
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.warning("Please provide Textract JSON data")

st.markdown("""
### How to use:
1. Choose input method (file upload or paste JSON)
2. Provide your Textract JSON output
3. (Optional) Customize the Index Name
4. Click 'Convert to Bulk Index Format'
5. Review the preview (optional)
6. Copy the output or download the file
7. Paste the content into OpenSearch Dev Tools console
""")

# Add AWS Textract direct integration info
st.sidebar.markdown("""
### About
This tool converts AWS Textract JSON output into OpenSearch bulk index format.

### Supported Features:
- Direct JSON file upload
- JSON text input
- Custom index naming
- Data preview
- Bulk index format download
""")

