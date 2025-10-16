from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def point_in_rectangle(x, y, 
                      x1,y1,x2,y2) -> bool:
    return x1 <= x <= x2 and y1 <= y <= y2
def simple_visualize_tracev2(trace_link):
    """改进的可视化TraceLink数据，避免边重叠并标记终止状态"""
    
    # 创建图
    G = nx.DiGraph()
    
    # 添加节点和边，记录终止状态
    terminal_states = set()
    for state in trace_link.states:
        state_name = Path(state.img_path).stem
        G.add_node(state_name)
        
        # 记录终止状态
        if state.is_done:
            terminal_states.add(state_name)
        
        # 添加转移边
        for action_type, actions in state.map_info.items():
            if actions:
                for key, target_path in actions.items():
                    if target_path:
                        target_name = Path(target_path).stem
                        G.add_edge(state_name, target_name, action=action_type)
    
    # 绘制图形
    plt.figure(figsize=(14, 10))
    
    # 按类别分组节点
    categories = {}
    for node in G.nodes():
        category = node.split('_')[0]  # 提取a部分
        if category not in categories:
            categories[category] = []
        categories[category].append(node)
    
    # 手动设置节点位置，相同类别的节点放在同一列
    sortcategories = sorted(categories.items(), key=lambda x: int(x[0]))
    pos = {}
    num_categories = len(categories)
    
    for i, (category, nodes) in enumerate(sortcategories):
        x = i * 2  # 增加列间距
        nodes_sorted = sorted(nodes, key=lambda x: int(x.split('_')[1]))  # 按b排序
        
        for j, node in enumerate(nodes_sorted):
            y = -j * 1.5  # 增加行间距
            pos[node] = (x, y)
    
    # 绘制节点（按类别着色 + 终止状态特殊标记）
    category_colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    color_map = {}
    
    for i, category in enumerate(categories.keys()):
        color_map[category] = category_colors[i]
    
    # 节点颜色和样式
    node_colors = []
    node_sizes = []
    node_edge_colors = []
    node_linewidths = []
    
    for node in G.nodes():
        category_color = color_map[node.split('_')[0]]
        node_colors.append(category_color)
        
        # 终止状态特殊标记
        if node in terminal_states:
            node_sizes.append(800)  # 更大的节点
            node_edge_colors.append('red')  # 红色边框
            node_linewidths.append(3)  # 更粗的边框
        else:
            node_sizes.append(500)
            node_edge_colors.append('black')
            node_linewidths.append(1)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes, 
                          node_color=node_colors, 
                          alpha=0.9, 
                          edgecolors=node_edge_colors,
                          linewidths=node_linewidths)
    
    # 绘制边（使用曲线避免重叠）
    edge_colors = []
    edge_styles = []
    edge_widths = []
    
    for u, v, data in G.edges(data=True):
        color_map = {'click': 'red', 'swipe': 'blue', 'input': 'green', 'wait': 'gray'}
        edge_colors.append(color_map.get(data['action'], 'black'))
        edge_styles.append('solid')
        edge_widths.append(1.5)
    
    # 使用曲线边
    nx.draw_networkx_edges(G, pos, 
                          edge_color=edge_colors, 
                          arrows=True, 
                          arrowsize=20, 
                          alpha=0.7,
                          arrowstyle='->',
                          connectionstyle="arc3,rad=0.1",  # 添加弧度避免直线重叠
                          style=edge_styles,
                          width=edge_widths)
    
    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # 绘制边标签（动作类型）
    edge_labels = {(u, v): data['action'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    # 添加图例
    from matplotlib.patches import Circle, Patch
    legend_elements = [
        Circle(0, color='red', label='done', fill=False, linewidth=3),
       # Patch(facecolor='lightgray', edgecolor='black', label='not done', linewidth=1),
    ]
    
    # 动作类型图例
    action_colors = {'click': 'red', 'swipe': 'blue', 'input': 'green', 'wait': 'gray'}
    for action, color in action_colors.items():
        legend_elements.append(
            Patch(facecolor=color, edgecolor='black', label=f'{action}')
        )
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title(f"FSM)", fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

    
    for category, nodes in categories.items():
        category_terminal = [node for node in nodes if node in terminal_states]
        print(f"  类别 {category}: {len(nodes)} 个状态 (其中终止状态: {len(category_terminal)})")
    
    # 打印终止状态列表
    if terminal_states:
        print("\n终止状态列表:")
        for state in sorted(terminal_states):
            print(f"  - {state}")