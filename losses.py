'''
  the author is leilei;
  Loss functions are in here.
  针对D loss 分别计算 有标签真实数据损失函数、生成数据损失函数、无标签真实数据损失函数。
  参照 https://blog.csdn.net/shenxiaolu1984/article/details/75736407
  G loss的话 就是简单 - Loss_fake
  
  这篇code作者分类导入标签和无标签数据
  没有把 标签生成数据损失函数loss 放进去，不然D 分类器就要k+1了， 0-k对应真实数据标签，k+1对应生成数据标签
'''

def log_sum_exp(x,axis=1):
    '''
    Args:
        x : [n*h*w,c],semantic segmentation‘s output’s shape is [n,c,h,w]，before input need to reshape [n*h*w,c]
    '''
    m = torch.max(x,dim=axis)[0]
    return m+torch.log(torch.sum(torch.exp(x-torch.unsqueeze(m,dim=axis)),dim=axis))

def Loss_label(pred,label): # 标签真实数据损失函数
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c] 
    label: [n,h,w] ,tensor need to numpy ,then need to reshape [n*h*w,1]
    '''
    shape = pred.shape# n c h w
    # predict before softmax
    output_before_softmax_lab = pred.transpose(1,2).transpose(2,3).reshape([-1,shape[1]]) # [n*h*w, c]
    
    label_ = label.data.cpu().numpy().reshape([-1,])
    # l_lab before softmax
    l_lab = output_before_softmax_lab[np.arange(label_.shape[0]),label_]
    # compute two value
    loss_lab = -torch.mean(l_lab) + torch.mean(log_sum_exp(output_before_softmax_lab))
    
    return loss_lab

def Loss_fake(pred): # 生成数据损失函数
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c] 
    '''
    shape = pred.shape# n c h w
    # predict before softmax
    output_before_softmax_gen = pred.transpose(1,2).transpose(2,3).reshape([-1,shape[1]])# [n*h*w, c]
    l_gen = log_sum_exp(output_before_softmax_gen)
    loss_gen = torch.mean(F.softplus(l_gen))
    
    return loss_gen

def Loss_unlabel(pred): # 无标签真实数据损失函数
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c] 
    '''
    shape = pred.shape# n c h w
    # predict before softmax
    output_before_softmax_unl = pred.transpose(1,2).transpose(2,3).reshape([-1,shape[1]])# [n*h*w, c]
    
    l_unl = log_sum_exp(output_before_softmax_unl)
    loss_unl = -torch.mean(l_unl) + torch.mean(F.softplus(l_unl))
    
    return loss_unl



