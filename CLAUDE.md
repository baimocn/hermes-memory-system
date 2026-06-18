# Claude 进化系统 - 全局配置

## 系统概述

这是一个增强Claude Code能力的进化系统，全局生效。

## ⚠️ 记忆召回规则（最高优先级）

你拥有一个长期记忆系统（vmem），它会在会话开始时自动注入近期记忆和教训。
但你也必须**主动查询**记忆系统，不要只写不读。

### 必须查询记忆的时机（不要等用户提醒）

| 场景 | 调用什么 |
|------|---------|
| 用户说"之前"、"上次"、"记得"、"那个" | `vmem_search(query)` |
| 遇到报错或异常 | `vmem_recall_lessons(query)` |
| 用户问偏好/习惯/风格 | `vmem_profile_get()` |
| 做技术决策前 | `vmem_search(query)` 查相关经验 |
| 修改 hooks/settings/mcp-servers | `vmem_recall_lessons(query="修改 <文件名>")` |
| 用户提到某个项目/工具名 | `vmem_search(query)` |

### 核心原则

```
记忆系统的价值 = 读取频率 × 读取质量

写 100 条记忆但从来不读 = 等于没有记忆
写 10 条记忆但每次都先查 = 真正的智能
```

**禁止行为：**
- ❌ 用户问"之前做过什么"时直接回答"我不记得"
- ❌ 遇到报错时不去查 lessons
- ❌ 做决策时不查相关记忆
- ❌ 只往 vmem_store 写，从不 vmem_search 读

**正确行为：**
- ✅ 每次回答前先想：这个场景下有没有相关记忆？
- ✅ 遇到问题先查 `vmem_recall_lessons`
- ✅ 用户提到模糊引用时主动搜索
- ✅ 搜到的记忆直接融入回答，不用告诉用户"我查了记忆"

## 使用方法

### 记忆工具

存储记忆：
```
请记住 <key> = <value>
```

检索记忆：
```
我之前记住了什么关于 <key> 的信息？
```

搜索记忆：
```
搜索关于 <关键词> 的记忆
```

### 学习工具

记录学习：
```
请记录这次经验：成功/失败 <描述>
```

获取学习建议：
```
有什么改进建议吗？
```

### 自动功能

**Hooks会自动执行：**
- 工具使用后自动记录学习
- 会话开始时自动加载上下文（画像 + 近期记忆 + 教训）
- 每条用户消息自动搜索相关记忆并注入（hermes-recall-hook）
- 会话结束时自动自省和复盘

## 文件位置

```
~/.claude/
├── .mcp.json                    # MCP服务器配置
├── settings.json                # 包含hooks配置
├── mcp-servers/
│   ├── vmem-server.py           # Hermes MCP服务器（12个工具）
│   └── vmem/
│       ├── store.py             # 核心存储（25+方法）
│       ├── migrations.py        # 13个schema迁移
│       └── config.py            # 路径自动检测
├── hooks/
│   ├── hermes-recall-hook.py    # UserPromptSubmit: 自动召回相关记忆
│   ├── hermes-preference-hook.py # UserPromptSubmit: 偏好提取
│   ├── hermes-profile-session-hook.py # SessionStart: 注入画像+记忆+教训
│   ├── hermes-error-hook.py     # PostToolUse: 错误→因果链
│   ├── hermes-introspect-hook.py # SessionEnd: 自省+GC
│   └── hermes-review-hook.py    # SessionEnd: 会话复盘
└── memory/
    └── vmem.db                  # 记忆数据库
```

## 注意事项

1. 需要重启Claude Code才能加载MCP服务器
2. Hooks会在后台自动执行，不会影响正常使用
3. 记忆数据存储在SQLite数据库中
4. 此文件位于用户主目录，所有项目都能读取
5. **记忆必须读写平衡** — 只写不读等于没有记忆
