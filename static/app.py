import streamlit as st
import requests

# Base URL of the FastAPI app
FASTAPI_BASE_URL = "https://insight-stream-service-kvbcxmn5bq-uc.a.run.app/" 

st.title('Insight Stream')

youtube_url = st.text_input('Enter YouTube URL')

if youtube_url:
    # POST request to the /analyzer endpoint
    analyze_response = requests.post(f"{FASTAPI_BASE_URL}/analyzer", json={"youtube_url": youtube_url})
    
    if analyze_response.status_code == 200:
        analysis_results = analyze_response.json()

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.video(youtube_url)
        
        with col2:
            st.write(analysis_results['summary'])
        
        st.subheader("Questions")
        questions_md = "\n".join([f"- {question}" for question in analysis_results['questions']])
        st.markdown(questions_md)
        
        st.subheader("Papers")
        papers_html = "<div style='height: 300px; overflow-y: auto; border: 1px solid #ccc;'>"
        papers_list = [f"<p><b>Title:</b> {paper['title']}<br></br><b>Abstract:</b> {paper['abstract']}</p>" for paper in analysis_results['papers']]
        papers_html += "".join(papers_list) + "</div>"
        st.markdown(papers_html, unsafe_allow_html=True)
        

        st.subheader("Query Interface")
                
        # Initialize the chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Ask me about the topics from the video. And, I can provide relevant research papers from arXiv!"}]
                
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Your question"): 

            st.session_state.messages.append({"role": "user", "content": prompt})
            full_response = ""

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # POST request to the /query endpoint
                    response = requests.post(f"{FASTAPI_BASE_URL}/query", json={"prompt": prompt})
                    
                    if response.status_code != 200:
                        response_text = "Failed to get a response from the server. Please try again."
                        st.session_state.messages.append({"role": "assistant", "content": "There was an error processing your request. Please try again."})
                    
                    elif response.status_code == 200:

                        response = response.json()
                        response_text = response['response']
                        papers = response['papers']

                        placeholder = st.empty()
                        full_response = f"{response_text}\n\n**Papers:**\n"

                        for paper in papers:
                            paper_details = f"- **Title:** {paper['title']}\n\n **Abstract:** {paper['abstract']}\n\n"
                            full_response += paper_details

                        placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
