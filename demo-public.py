import requests
import streamlit as st


# 初始化数据存储
if 'result' not in st.session_state:
    st.session_state.result = None

# 页面标题
st.title("污泥产物预测系统")

# 创建输入框
input1 = st.text_input("污泥添加比例（%）", key="input1")
input2 = st.text_input("C含量（%）", key="input2")
input3 = st.text_input("热解温度（℃）", key="input3")

# 创建按钮
if st.button("预测"):
    with st.spinner(" 预测中..."):
        # 按钮点击后的处理逻辑
        if input1 and input2 and input3:
            data = {"污泥添加比例（%）": input1, "C含量（%）": input2, "热解温度（℃）": input3}
            req = requests.post('http://localhost:5000/predict', data=data)
            st.session_state.result = req.json()
            # st.success(req)
        else:
            st.warning(" 请填写输入框")

# 结果展示区域
if st.session_state.result:
    st.success(" 数据预测完成！")

    # 指标卡片
    # cols = st.columns(7)
    # cols[0].metric("气体中CH4含量（%）", st.session_state.result["predictions"]["气体中CH4含量（%）"])
    # cols[1].metric("气体中CO2含量（%）", st.session_state.result["predictions"]["气体中CO2含量（%）"])
    # cols[2].metric("气体产率（%）", st.session_state.result["predictions"]["气体产率（%）"])
    # cols[3].metric("液体产率（%）", st.session_state.result["predictions"]["液体产率（%）"])
    # cols[4].metric("热解油中含氮化合物含量（%）", st.session_state.result["predictions"]["热解油中含氮化合物含量（%）"])
    # cols[5].metric("热解油中酚含量（%）", st.session_state.result["predictions"]["热解油中酚含量（%）"])
    # cols[6].metric("热解油中酸含量（%）", st.session_state.result["predictions"]["热解油中酸含量（%）"])

    # 样式一
    # st.metric("气体中CH4含量（%）", st.session_state.result["predictions"]["气体中CH4含量（%）"])
    # st.metric("气体中CO2含量（%）", st.session_state.result["predictions"]["气体中CO2含量（%）"])
    # st.metric("气体产率（%）", st.session_state.result["predictions"]["气体产率（%）"])
    # st.metric("液体产率（%）", st.session_state.result["predictions"]["液体产率（%）"])
    # st.metric("热解油中含氮化合物含量（%）", st.session_state.result["predictions"]["热解油中含氮化合物含量（%）"])
    # st.metric("热解油中酚含量（%）", st.session_state.result["predictions"]["热解油中酚含量（%）"])
    # st.metric("热解油中酸含量（%）", st.session_state.result["predictions"]["热解油中酸含量（%）"])

    # 原始数据查看
    # with st.expander(" 查看原始JSON数据"):
    #     st.json(st.session_state.result)

    # 样式二-纯markdown
    # st.markdown("###  预测结果")
    # for key, value in st.session_state.result["predictions"].items():
    #     st.markdown(f"**{key.title()}**:  {value}")

    # 样式三-markdown + HTML
    st.markdown("###  预测结果", unsafe_allow_html=True)
    for key, value in st.session_state.result["predictions"].items():
        st.markdown(f"<div  style='margin: 5px 0;color: green'><strong>{key}:</strong> {value}</div>",
                    unsafe_allow_html=True)

    # st.write("###  原始结果")
    # st.write(st.session_state.result["predictions"])