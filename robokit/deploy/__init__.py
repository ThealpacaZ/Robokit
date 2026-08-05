"""真机部署运行时：模型登记表、公共运行时、两条执行循环。

入口脚本：
    scripts/serve_policy.py   GPU 服务端（sync / RTC 两种服务共用一个入口）
    scripts/run_policy.py     真机端（sync / RTC 两条循环共用一个入口）
"""
