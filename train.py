# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GicLoss, GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from tensorboardX import SummaryWriter

from visualization import board_add_image, board_add_images, save_images


def get_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name", default="GMM") # tên huấn luyện
    
    parser.add_argument("--gpu_ids", default="") # GPU sẽ được dùng, nếu để trống thì dùng CPU
    
    parser.add_argument('-j', '--workers', type=int, default=1)# số lượng worker được sử dụng để đọc dữ liệu
    
    parser.add_argument('-b', '--batch-size', type=int, default=4)# Kích thước batch

    parser.add_argument("--dataroot", default="data") # vị trí dataset

    parser.add_argument("--datamode", default="train") # kiểu chạy, mặc định là huấn luyện

    parser.add_argument("--stage", default="GMM") # giai đoạn của mô hình mặc định là GMM
    
    # parser.add_argument("--stage", default="TOM")

    parser.add_argument("--data_list", default="train_pairs.txt") # file danh sách chứa tập tin dữ liệu
    
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')# thư mục luu kết qủa huấn luận
    
    parser.add_argument("--fine_width", type=int, default=192)# kích thước ảnh with 192
    
    parser.add_argument("--fine_height", type=int, default=256)# kích thước ảnh height 256
    
    parser.add_argument("--radius", type=int, default=5)#bán kinh của vùng lặp ảnh (TOM)
    
    parser.add_argument("--grid_size", type=int, default=5)#kích thước lưới (TOM)
    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam') #Tốc độ học ban đầu cho thuật toán tối ưu Adam
    
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='Lưu thông tin TensorBoard')
    
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='Lưu thông tin checkpoint')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Model checkpoint để khởi tạo')
    
    parser.add_argument("--display_count", type=int, default=20) # hiển thị kết quả qua số bước(20)
    
    parser.add_argument("--save_count", type=int, default=5000) #lưu kết quả qua số bước (5000)
    
    parser.add_argument("--keep_step", type=int, default=100000) # Số lần chạy (điều kiện dừng)
    
    parser.add_argument("--decay_step", type=int, default=100000) # Số lần để giảm tốc độ học (learning rate decay)
    # Cơ chế giảm tốc độ học (learning rate decay) thường được sử dụng để giảm kích thước bước học (learning rate) theo thời gian
    
    parser.add_argument("--shuffle", action='store_true',
                        help='Trộn dữ liệu đầu vào')
    
    # Parse các tham số từ dòng lệnh và trả về kết quả
    opt = parser.parse_args()
    return opt

# hàm GMM
# hàm thực hiện code không sử dụng cuda
def train_gmm(opt, train_loader, model, board):
    model.cpu()  # Chuyển mô hình ra khỏi GPU
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    
    gicloss = GicLoss(opt)
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))
        
    # số lần huấn luyện bằng keep_step + decay_step (100k+100k)=> qúa trình huấn luyện pải chạy 200k step để hoàng thành 
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        
        #Xử lý dữ liệu đầu vào: train_loader là cấu hình dữ liệu trong CDdataset from cp_dataset
        inputs = train_loader.next_batch()
        
        cloth_tensor = inputs['cloth']  # lấy mẫu vải trong folder cloth
        
        
        # setup thực hiện lọc ngững biến vãi màu trong folder cloth thành ảnh mark(cloth_mark)
        gray_cloth_mask = torch.mean(cloth_tensor, dim=1, keepdim=True) # thay đổi kênh màu từ ( 3 -> 1)
        
        # thiết lặp vị trí dữ liệu 
        cloth_array = gray_cloth_mask.permute(0, 1, 2, 3).numpy() #permute(weight, chanel color, width, height)
        
        # Chuyển đổi NumPy array thành hình ảnh OpenCV
        cloth_mask = (cloth_array * 255).astype(np.uint8)  # Điều chỉnh giá trị để chuyển sang 8-bit
        
        # Chuyển sang không gian màu RGB
        hsv = cv2.cvtColor(cloth_mask, cv2.COLOR_BGR2RGB)
        
        # Tạo mask cho các pixel vượt qua ngưỡng và chuyển chúng thành màu đen
        threshold = 250  # Ngưỡng màu trắng thuần(240), được sửa thành 250 để xác định màu trắng rõ hơn
        
        mask = cv2.inRange(hsv, (threshold, threshold, threshold), (255, 255, 255)) 
        
        cloth_mask[mask > threshold] = 0  # Chuyển các pixel vượt qua ngưỡng thành màu đen
        cloth_mask[mask <= threshold] = 255  # Chuyển các pixel không vượt qua ngưỡng thành màu trắng
        
        
        # Chuyển dữ liệu ra khỏi GPU
        im = inputs['image'].cpu()  # ảnh người mẫu và là ảnh dùng để train
        
        im_pose = inputs['pose_image'].cpu()  # chứa dáng người xác định vị trí (đầu, vai , tay , chân) được xử dụng để xữ lý ảnh
        
        im_h = inputs['head'].cpu()  # ảnh đầu vào chứa thông tin về phần đầu của người mẫu
        
        shape = inputs['shape'].cpu()  # Dữ liệu này có thể liên quan đến hình dạng cơ thể, 
        #có thể bao gồm thông tin về chiều cao, cân nặng, hoặc các thuộc tính hình thể khác.
        
        agnostic = inputs['agnostic'].cpu()  
        
        c = inputs['cloth'].cpu()  # ảnh vải trong quá trình huấn luyện được đặt lên cở thể người
        
        # cm = inputs['cloth_mask'].cpu()  #mặt nạ của vải, được sử dụng để chỉ ra vị trí vải trên hình ảnh.
        
        cm = torch.from_numpy(cloth_mask.astype(np.float32)).cpu()
        
        im_c = inputs['parse_cloth'].cpu()  
        
        im_g = inputs['grid_image'].cpu()  #  lưới
        
        grid, theta = model(agnostic, cm)    # Có thể thêm c vào cho việc huấn luyện mới
        
        # Chuyển dữ liệu lên GPU và tính toán loss:
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        # tập dữ liêu sau khi train  được đưa vào mảng visuals
        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth+im)*0.5, im]]
        
        # Loss cho warped cloth
        Lwarp = criterionL1(warped_cloth, im_c) # đoạn này tính gía trị los của hình ảnh 
        # Hàm loss thực tế như trong bài báo dưới đây (comment dòng trước và uncomment dòng sau để huấn luyện theo bài báo)
        # Lwarp = criterionL1(warped_mask, cm)
        # Loss chính quy hóa grid 
        # gọi grid để tính toán độ lệch giữa các cell
        Lgic = gicloss(grid)
        # 200x200 = 40.000 * 0.001
        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])
        # tính tổng giá trị thất thoát bao gồm L1 lấy từ (warped_cloth, im_c) và grid los 
        loss = Lwarp + 40 * Lgic    # Tổng loss GMM
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
         # Hiển thị thông tin và giám sát trong quá trình huấn luyện
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1) # thêm dữ liệu img vào tensorboard
            board.add_scalar('loss', loss.item(), step+1) # thêm số liệu loss vào sơ đồ tensorboard
            board.add_scalar('40*Lgic', (40*Lgic).item(), step+1) #
            board.add_scalar('Lwarp', Lwarp.item(), step+1) #
            
            # điều chỉnh in ra màng hình kết quả huấn luyện
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f, (40*Lgic): %.8f, Lwarp: %.6f' %
                  (step+1, t, loss.item(), (40*Lgic).item(), Lwarp.item()), flush=True)
                 # Lưu ảnh vào thư mục kết quả
                 
        # lưu kết quả check point  vào đường dẫn chỉ định
        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
            
            
def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()# Định nghĩa hàm loss dựa trên sự khác biệt giữa hai tensor bằng norm L1 (Mean Absolute Error). 
    # Hàm này được sử dụng để tính loss cho việc so sánh giữa ảnh đã biến đổi
    #(warped_cloth) và ảnh mục tiêu (im_c).
    # l1 được tính bằng cách lấy tổng các giá trị vector trong không gian vd  |x1| + |x2| + ... + |xn|.
    #chuẩn hoá L1 được sử dụng để tính toán sự khác biệt giữa hai tensor  ( sự sai lệch giữa dự đoán và mục tiêu )
   
    criterionVGG = VGGLoss()# định nghĩa hàm VGGlosss
   
    criterionMask = nn.L1Loss()

    # optimizer
    #  model. Adam là một thuật toán tối ưu hóa thường được sử dụng để điều chỉnh trọng số của mạng neural trong quá trình huấn luyện.
    optimizer = torch.optim.Adam(
        # lr=opt.lr: Đây là tốc độ học
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
    
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        pcm = inputs['parse_cloth_mask'].cuda()

        # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
        outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        """visuals = [[im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]"""  # CP-VTON

        visuals = [[im_h, shape, im_pose],
                   [c, pcm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]  # CP-VTON+

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        # loss_mask = criterionMask(m_composite, cm)  # CP-VTON
        loss_mask = criterionMask(m_composite, pcm)  # CP-VTON+
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step+1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)
            
        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(
                opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
            
            
            
def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset) # lấy dữ liệu từ dataset

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))


    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
            
         # 3 dữ liệu vào vào (cấu hình huấn luyện opt, dữ liệu lấy từ dataset, model thuật toán lấy từ network, tensorboard dùng để luu kết quả)    
        train_gmm(opt, train_loader, model, board)
        
        
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        model = UnetGenerator(
            26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
