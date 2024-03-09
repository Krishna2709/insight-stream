import streamlit as st
from insight import VideoAnalysis
from insight import chat_engine

# Start of the Streamlit webapp
st.title('YouTube Video Analysis Webapp')

# YouTube video link input
youtube_url = st.text_input('Enter YouTube video URL')

if youtube_url:
    video_analysis = VideoAnalysis()
    analysis_results = video_analysis.analyze(youtube_url)

    # Top section for video and summary
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.video(youtube_url)
    
    with col2:
        st.write(analysis_results['summary'])
    

    st.subheader("Questions")
    questions_md = "\n".join([f"- {question}" for question in analysis_results['questions']])
    st.markdown(questions_md)
    
    
    st.subheader("Papers")
    # Creating a scrollable section for papers using custom HTML and CSS
    papers_html = "<div style='height: 300px; overflow-y: auto; border: 1px solid #ccc;'>"
    papers_list = [f"<p><b>Title:</b> {paper['title']}<br></br><b>Abstract:</b> {paper['abstract']}</p>" for paper in analysis_results['papers']]
    papers_html += "".join(papers_list) + "</div>"
    st.markdown(papers_html, unsafe_allow_html=True)
    
    
    st.subheader("Chat Interface")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you?"}]

    user_prompt = st.chat_input("Your question")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        response_text = chat_engine(user_prompt, video_analysis.youtube_engine, video_analysis.vector_engine, video_analysis.llm)

        # Assuming the response from chat_engine is a string. Adjust if it's different.
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # If last message is not from assistant, generate a new response
        if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
            prompt = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = video_analysis.chat_engine(prompt)
                    response_text = response # Adjust based on the actual response structure
                    st.write(response_text)
                    message = {"role": "assistant", "content": response_text}
                    st.session_state.messages.append(message)  # Add response to message history
