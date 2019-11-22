# imports
import os
import torch
import numpy as np
import img_tools as im_t
import matplotlib.pyplot as plt
import networks.functional as f
from sklearn.metrics import auc
from image_codec import ImageCodec
from torchvision.utils import make_grid
from img_tools import EvaluationImageDataLoaders, InvNormalization


# -----------------------------------------------------------------------------------------------------------------------
# Comparison of Models Class
# -----------------------------------------------------------------------------------------------------------------------

# constant variables
SAVED_MODELS_PATH = "./saved_models/"
SAVED_CODECS_PATH = "./image_codec/saved_cc/"

"""
CompareModels
 
 Compares performance various Autoencoder Models
         
         Args:
            img_dir (String)             : path to image directory
            models  (list of nn.Modules) : list of trained models to compare
            codecs  (list of String)     : list of standard lossy codec names 
            dataset (string)             : dataset on which to perform evaluation ['test', 'valid']

"""


class CompareModels:

    def __init__(self, img_dir, models, codecs):

        # Image Directory
        img_dir = os.path.expanduser(img_dir)

        if os.path.isdir(img_dir):
            self.img_dir = img_dir
        else:
            raise NotADirectoryError(
                img_dir + " is not a directory"
            )

        # List Standard Lossy Codecs
        self.codecs = codecs

        # List Deep Image Compression Models
        self.models = models

    def display_compression_curve(self, metric, dataset='valid'):

        # plot compression curves

        # setup plot
        im_t.setup_plot(
            title='',
            y_label=metric,
            x_label="bits-per-pixel (bpp)"
        )

        legend = []

        fmt_index = 0
        fmts = ['o--g', 'x--r', '^--b', '*--c']

        # plot for Standard Image Codecs
        for codec in self.codecs:
            legend.append(codec)

            # load compression curve
            curve = ImageCodec(self.img_dir, codec).load_cc(
                metric=metric,
                save_dir=SAVED_CODECS_PATH + dataset
            )

            # plot rate distortion curve
            plt.plot(curve['bpp'], curve['met'], fmts[fmt_index])
            fmt_index += 1

        # plot for Deep Compression Models

        for model in self.models:
            legend.append(model.display_name)

            # load compression curve
            curve = EvaluateModel(self.img_dir, model).load_compression_curve(
                save_dir=SAVED_MODELS_PATH + model.name + '/' + dataset,
                metric=metric
            )

            # plot rate-distortion curve
            plt.plot(curve['bpp'], curve['met'], fmts[fmt_index])
            fmt_index += 1

        plt.legend(legend, loc='lower right', fontsize='large')
        plt.savefig('./' + metric + '_curve.pdf')
        plt.show()

        return

    def display_compressed_images(self, itrs, start_itr=0, dataset='valid'):

        # plot progressive images

        for model in self.models:
            EvaluateModel(
                self.img_dir + "/" + dataset,
                model
            ).progressive_imshow(itrs, start_itr)

        for codec in self.codecs:
            ImageCodec(
                self.img_dir + "/" + dataset,
                codec
            ).progressive_imshow(itrs, start_itr)

        return

    def display_auc(self, metric, dataset, bpp_max=2.0, bpp_min=0.0):

        # display Area Under Curve
        print("Displaying AUC:")
        print("\nDeep Compression Models")

        for model in self.models:

            # load compression curve
            curve = EvaluateModel(self.img_dir, model).load_compression_curve(
                metric=metric,
                save_dir=SAVED_MODELS_PATH + model.name + '/' + dataset
            )

            cut_curve = {'bpp': [], 'met': []}
            for i in range(len(curve['bpp'])):
                if bpp_min <= curve['bpp'][i] <= bpp_max:
                    # save values in range
                    cut_curve['bpp'].append(curve['bpp'][i])
                    cut_curve['met'].append(curve['met'][i])

            curve_area = auc(cut_curve['bpp'], cut_curve['met'])

            print("{} : {}".format(model.display_name, curve_area))

        print("\nStandard Codecs")

        # plot compression curves for standard image codecs
        for codec in self.codecs:

            # load compression curve
            curve = ImageCodec(self.img_dir, codec).load_cc(
                metric=metric,
                save_dir=SAVED_CODECS_PATH + dataset
            )

            cut_curve = {'bpp': [], 'met': []}

            for i in range(len(curve['bpp'])):

                if bpp_min <= curve['bpp'][i] <= bpp_max:
                    # save values in range
                    cut_curve['bpp'].append(curve['bpp'][i])
                    cut_curve['met'].append(curve['met'][i])

            # interpolate last point
            cut_curve['met'].append(
                (curve['met'][-1] - cut_curve['met'][-1]) / curve['bpp'][-1] * (2.0 - cut_curve['bpp'][-1]) +
                cut_curve['met'][-1]
            )
            cut_curve['bpp'].append(2.0)

            curve_area = auc(cut_curve['bpp'], cut_curve['met'])

            print("{} : {}".format(codec, curve_area))

        return

    
# ----------------------------------------------------------------------------------------------------------------------
# Single Model Evaluation Class
# ----------------------------------------------------------------------------------------------------------------------

"""
Class EvaluateModel:

    Various methods used to evaluate an Autoencoder model (full iteration evaluations)
    
        Args:
            model (nn.Module) : trained model to be evaluated
"""


class EvaluateModel:
    
    def __init__(self,
                 img_dir,
                 model, img_s=(224, 320), intrinsic=False):
        
        # use GPU if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Inference mode and on device
        self.model = model.to(self.device)
        self.model.train(False)

        if model.name in ["SINet", "MaskedBINet", "BINetAR", "BINetOSR"]:
            self.p_s = self.model.g_s
        else:
            self.p_s = self.model.p_s

        img_dir = os.path.expanduser(img_dir)

        if not os.path.isdir(img_dir):
            raise NotADirectoryError("Specified image directory d.n.e!")
        else:
            self.img_dir = img_dir

        # patch evaluation
        self.intrinsic = intrinsic

        # perform arithmetic coding
        self.img_h, self.img_w = img_s

        # define Evaluation dataLoader
        eval_dls = EvaluationImageDataLoaders(
            img_dir=img_dir,
            img_s=img_s,
            p_s=self.p_s,
            b_s=1
        )

        # Inverse Normalization Transform [-1, 1] -> [0, 1]
        self.inv_norm = InvNormalization(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
        
        # Image DataLoader
        self.img_dl = eval_dls.get_img_dl()
        
        # Patch DataLoader
        eval_dls.b_s = 1
        self.patch_dl = eval_dls.get_patch_dl()

    def apply_model(self, inpt, itrs=None):
        # compress input using model
        with torch.no_grad():
            # run in inference mode
            if self.model.name in ["SINet", "MaskedBINet"]:
                output = self.model.encode_decode(
                    inpt.to(self.device)
                )
            else:
                output = self.model.encode_decode(
                    inpt.to(self.device),
                    itrs=itrs
                )
        return output
                
    def _progressive_eval(self, metric):

        # calculate average quality at each model itr

        # metric values
        m_val = []
        
        for i in range(self.model.itrs):
            # calculate & append metric value
            if self.intrinsic:
                # evaluation on center patches
                m_val.append(self._average_patch_eval(metric, i+1))
            else:
                # evaluation on full-scale images
                m_val.append(self._average_img_eval(metric, i+1))
            
        return m_val

    def _average_patch_eval(self, metric, itrs=None, print_stat=False):
        # calculate patch quality at specific iteration

        m_val = []
        p_h, p_w = self.model.p_s

        for r_patch in self.patch_dl:

            # encode and decode patch
            c_patch = self.apply_model(r_patch, itrs=itrs)

            # inverse normalisation
            if self.model.name in ["BINetAR", "BINetOSR", "SINet", "MaskedBINet"]:
                r_patch = self.inv_norm(r_patch[0][:, p_h: -p_h, p_w: -p_w])
                c_patch = self.inv_norm(c_patch[0][:, p_h: -p_h, p_w: -p_w].cpu())
            else:
                r_patch = self.inv_norm(r_patch[0])
                c_patch = self.inv_norm(c_patch[0].cpu())

            if metric == "SSIM":
                # calculate SSIM
                m_val.append(self._evaluate_ssim(r_patch, c_patch))

            elif metric == "PSNR":
                # calculate PSNR
                m_val.append(self._evaluate_psnr(r_patch, c_patch))

        # average metric value
        m_avg = sum(m_val) / len(m_val)

        if print_stat:
            # print average metric
            print("Average " + metric + " over {} Images:".format(len(m_val)))
            print(str(m_avg))
            return

        return m_avg

    def _average_img_eval(self, metric, itrs, print_stat=False):
        # calculate average quality at specific itr

        # metric scores
        m_val = []

        for r_img in self.img_dl:

            # compress image
            c_img = self.apply_model(r_img, itrs)

            # [-1, 1] -> [0, 1]
            r_img = self.inv_norm(r_img[0])
            c_img = self.inv_norm(c_img[0].cpu())

            # evaluate metric
            if metric == "SSIM":
                m_val.append(
                    self._evaluate_ssim(r_img, c_img)
                )

            elif metric == "PSNR":
                m_val.append(
                    self._evaluate_psnr(r_img, c_img)
                )
                
        # average metric value
        m_avg = sum(m_val) / len(m_val)
        
        if print_stat:
            # print average metric
            print("Average " + metric + " over {} Images:".format(len(m_val)))
            print(str(m_avg))
     
        return m_avg
    
    def _calc_compression_curve(self, metric, save_dir):

        # calculate and save rate-distortion curve

        # bpp axis
        bpp = self.model.b_n / (self.model.p_w * self.model.p_h)
        bpp = np.linspace(bpp, bpp * self.model.itrs, self.model.itrs)
        
        # calculate metric values
        m_val = self._progressive_eval(metric)
        
        # compression curve dictionary
        curve = {'bpp': bpp, 'met': m_val}
        
        # create file name
        file_name = "".join([
            save_dir,
            '/',
            self.model.name,
            '_',
            metric,
            '.npy'
        ])

        # save curve as numpy file
        np.save(file_name, curve)
 
        return
        
    def load_compression_curve(self, metric, save_dir):

        if metric not in ["SSIM", "PSNR"]:
            raise KeyError("Specified metric : {} is not currently supported!".format(metric))

        save_dir = os.path.expanduser(save_dir)

        if not os.path.isdir(save_dir):
            raise NotADirectoryError("Specified directory d.n.e!")
        
        # create filename
        file_name = "".join([
            save_dir,
            '/',
            self.model.name,
            '_',
            metric,
            '.npy'
        ])
        
        if not os.path.isfile(file_name):
            # calculate compression curve
            print("Creating Compression Curve File : ")
            self._calc_compression_curve(metric, save_dir)

        # load curve
        curve = np.load(file_name).item()
            
        return curve
    
    def compare_img(self, itrs):
        # display reference and compressed image

        r_img = iter(self.img_dl).next()
        c_img = self.apply_model(r_img, itrs)

        # Inverse Normalization [-1,1] -> [0,1]
        r_img = self.inv_norm(r_img[0])
        c_img = self.inv_norm(c_img[0].cpu())

        # display images
        self._display_images(r_img, c_img)
        
        return
    
    def save_img(self, itrs):
        # save compressed image

        r_img = iter(self.img_dl).next()
        c_img = self.apply_model(r_img, itrs)

        # [-1,1] -> [0,1]
        c_img = self.inv_norm(c_img[0].cpu())
        
        # save image
        im_t.save_img(c_img)

        return
    
    def compare_patches(self, itrs=None):
        # display reference & compressed patches side by side
        
        # fetch patches
        r_patches = iter(self.patch_dl).next()

        # compress & reconstruct patches
        c_patches = self.apply_model(r_patches, itrs)

        if self.model.name in ["BINetAR", "BINetOSR"]:
            # only compare center patch
            r_patches = r_patches[:, :, self.model.p_h: -self.model.p_h, self.model.p_w: -self.model.p_w]
            c_patches = c_patches[:, :, self.model.p_h: -self.model.p_h, self.model.p_w: -self.model.p_w]
        elif self.model.name in ["MaskedBINet", "SINet"]:
            # only compare center patch
            r_patches = r_patches[:, :, self.model.p_h: -self.model.p_h, self.model.p_w: -self.model.p_w]

        # group patches into grids for display
        r_patches = make_grid(r_patches, nrow=self.patch_dl.batch_size, padding=1)
        c_patches = make_grid(c_patches, nrow=self.patch_dl.batch_size, padding=1)

        # [-1,1] -> [0,1]
        r_patches = self.inv_norm(r_patches)
        c_patches = self.inv_norm(c_patches.cpu())

        # display patches
        self._display_images(r_patches, c_patches)

        return

    def _print_evaluation(self, ref, comp):
        
        # print evaluation metrics

        # SSIM
        ssim = self._evaluate_ssim(ref, comp)
        print("SSIM : {}".format(ssim))
        
        # PSNR
        psnr = self._evaluate_psnr(ref, comp)
        print("PSNR : {}".format(psnr))
        
        return

    @staticmethod
    def _evaluate_ssim(ref, comp):
        #  calculate SSIM
        return im_t.EvalMetrics.SSIM.calc(ref, comp)

    @staticmethod
    def _evaluate_psnr(ref, comp):
        # calculate PSNR
        return im_t.EvalMetrics.PSNR.calc(ref, comp)

    def _display_images(self, ref, comp):
        # display reference & compressed images
        im_t.vs_imshow(ref, comp)
        self._print_evaluation(ref, comp)
        return
    
    def progressive_imshow(self, itrs, start_itr=0, widget=False):
        # display progressive image enhancement

        if self.model.name in ["SINet", "MaskedBINet"]:
            # sanity check
            print("Specified model is not progressive!")
            return

        bpp = []
        ssim = []
        psnr = []
        c_imgs = []
        
        # get input image
        r_img = iter(self.img_dl).next()

        for i in range(start_itr, itrs, 1):
            
            # append bpp
            bits = self.model.b_n * (i+1) / (self.model.p_w * self.model.p_h)
            bpp.append(bits)

            # compress image
            c_img = self.apply_model(r_img, i+1)

            # [-1,1] -> [0,1]
            r_img_inv = self.inv_norm(r_img[0])
            c_img_inv = self.inv_norm(c_img[0].cpu())

            # calculate SSIM
            ssim.append(
                round(self._evaluate_ssim(r_img_inv, c_img_inv), 2)
            )

            # calculate PSNR
            psnr.append(
                round(self._evaluate_psnr(r_img_inv, c_img_inv), 2)
            )

            c_imgs.append(c_img_inv)

        # stack compressed images
        c_imgs = torch.stack(c_imgs, dim=0)

        if widget is True:
            # display widget
            im_t.disp_images_widget(
                title=self.model.display_name,
                imgs=c_imgs,
                bpp=bpp,
                psnr=psnr,
                ssim=ssim
            )
            
        else:
            # display images & bpp
            im_t.disp_prog_imgs(
                title=self.model.display_name,
                imgs=c_imgs,
                bpp=bpp,
                psnr=psnr,
                ssim=ssim,
                start_itr=start_itr
            )
        
        return

    def display_inpainting(self, save=False):

        if self.model.name in ["ConvAR", "ConvRNN"]:
            # sanity check
            print("Specified model is not an inpainting network!")
            return

        # model patch height & width
        p_h, p_w = self.model.p_s

        # context region, ground truth, inpainting
        context = iter(self.patch_dl).next()
        inpaint = self.apply_model(context, itrs=1)[0].cpu()
        g_truth = context[0][:, p_h:-p_h, p_w:-p_w].contiguous()

        if self.model.name in ["MaskedBINet", "SINet"]:
            # mask center patch
            context[:, :, p_h:-p_h, p_w:-p_w] = -1.0
        if self.model.name in ["SINet"]:
            # white out patches not in context
            context[:, :, -p_h:] = 1.0
            context[:, :, p_h:-p_h, -p_w:] = 1.0

        context = f.sliding_window(context, p_h, p_w)
        context = make_grid(
            context,
            nrow=3,
            padding=1,
            pad_value=1,
            normalize=True
        )

        inpaint = self.inv_norm(inpaint)
        g_truth = self.inv_norm(g_truth)

        # display inpainting
        im_t.display_inpaint(context, inpaint, g_truth)

        if save:
            # save images
            im_t.save_img(context, save_loc="./context.png")
            im_t.save_img(inpaint, save_loc="./inpaint.png")
            im_t.save_img(g_truth, save_loc="./g_truth.png")

        return

