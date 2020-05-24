import streamlit as st


def main():
    st.title("How big should my language model be ?")
    gpu = st.radio("GPU", ("V100", "K80", "P4", "P100"))
    t_slider = st.empty()
    budget_slider = st.empty()
    flo_slider = st.empty()
    t = t_slider.slider('Wall time')
    budget = budget_slider.slider('Total cost (GCP)')
    flo = flo_slider.slider('Floating-point operations (FLOs)')
    flo = t



if __name__ == "__main__":
    main()
