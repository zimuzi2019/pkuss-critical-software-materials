## 使用方法

1. 安装依赖，下载嵌入模型`m3e-base`

   ```powershell
   pip install -r requirements.txt
   git clone https://huggingface.co/moka-ai/m3e-base
   ```

2. 如有必要，检查`configs\model_config.py`中`m3e-base`的路径是否正确

3. 运行

   ```
   python launch.py -a
   ```


4. 如果想要使用本地大模型`ChatGLM2-6B`，执行

   ```
   git clone https://huggingface.co/THUDM/chatglm2-6b
   ```

   克隆到本地，并在`configs\model_config.py`修改路径