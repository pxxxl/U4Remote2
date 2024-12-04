def compare_json_structure(obj1, obj2):
    """
    递归检查两个 JSON 对象的结构是否完全一致，只关注 dict 的键结构。

    :param obj1: 第一个 JSON 对象
    :param obj2: 第二个 JSON 对象
    :return: 如果结构一致返回 True，否则返回 False
    """
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        # 检查两个字典的键是否一致
        if set(obj1.keys()) != set(obj2.keys()):
            return False
        
        # 递归检查字典中的每个键对应的值
        for key in obj1:
            if not compare_json_structure(obj1[key], obj2[key]):
                return False
        return True
    
    # 如果是列表，跳过内部检查，只验证它们是否都是列表
    elif isinstance(obj1, list) and isinstance(obj2, list):
        return True
    
    # 如果不是字典或列表，则认为结构一致
    else:
        return True
