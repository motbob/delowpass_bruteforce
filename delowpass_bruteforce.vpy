import vapoursynth as vs
from vapoursynth import core
from vstools import depth, initialize_clip, get_y, get_u, get_v
from operator import itemgetter
import math
from vskernels import Blackman, Welch, WindowedSinc
from typing import Any
from math import pi, cos, sin, sqrt, log, exp
from vsview import set_output, is_preview
import os

def bohman_window(x: float, radius: int) -> float:
    return (1 - abs(x) / radius) * cos(pi*abs(x) / radius) + (1/pi)*sin(pi*abs(x) / radius)
class Bohman(WindowedSinc):
    def __init__(self, taps: float = 4, **kwargs: Any) -> None:
        super().__init__(taps, bohman_window, **kwargs)

def delowpass_bruteforcer(frame_numbers, clean_8, lowpassed_8, ll, bits, kernels, plane, clamp_list, chroma_direction, upper_taps_limit, upper_blur_limit):
    report_clamp = False
    if frame_numbers == [0]: #if the input is an image file, we must clamp, and we must not tell the user that clamping the actual delowpass is appropriate
        clamp_list = [True]
    elif len(clamp_list) == 2:
        report_clamp = True
    if ll == [False]:
        report_ll = False
    else:
        report_ll = True
    if bits == [16]:
        report_bits = False
    else:
        report_bits = True
    lowest_error = 100
    lower_taps_limit = 2
    taps_step = 0.01666666
    error_list = []
    if plane == 0:
        upper_blur_limit = math.ceil((upper_blur_limit - 1) / taps_step)
    else:
        upper_blur_limit = math.ceil((upper_blur_limit - 1) * 2 / taps_step)
    frame_number = frame_numbers[0]
    clean_8_n = clean_8[frame_number]
    lowpassed_8_n = lowpassed_8[frame_number]
    clean_16 = depth(clean_8_n, 16)
    for clamp_bool in clamp_list:
        print(f"Checking clamp = {clamp_bool}")
        for j in bits:
            if j == 8:
                clean_1 = clean_8_n
            else:
                clean_1 = clean_16
            for i in ll:
                if i:
                    if plane != 0:
                        continue
                    if j == 8:
                        clean_2 = core.resize.Point(core.resize.Lanczos(clean_1, format=vs.YUV444P8), transfer=8)
                    else:
                        clean_2 = core.resize.Point(core.resize.Lanczos(clean_1, format=vs.YUV444P16), transfer=8)
                else:
                    clean_2 = clean_1
                for t in range(lower_taps_limit, upper_taps_limit + 1):
                    print(f"Checking {t} taps...")
                    for b in range(1, upper_blur_limit):
                        for kernel in kernels:
                            if kernel == "bilinear":
                                if t == lower_taps_limit:
                                    if plane == 0:
                                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=kernel, taps=[1, 2], fv=[1 / (1 + b * taps_step), 1], fh=[1 / (1 + b * taps_step), 1])
                                    elif chroma_direction == "h":
                                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=kernel, taps=[2, 1], fv=[1, 1 / (1 + b * taps_step)], fh=[1, 1])
                                    elif chroma_direction == "w":
                                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=kernel, taps=[2, 1], fv=[1, 1], fh=[1, 1 / (1 + b * taps_step)])
                                else:
                                    continue
                            elif kernel=="welch":
                                clean_mangled = Welch(blur=(1 + b * taps_step), taps=t).scale(clean_2, force=True)
                            elif kernel=="bohman":
                                clean_mangled = Bohman(blur=(1 + b * taps_step), taps=t).scale(clean_2, force=True)
                            else:
                                if plane == 0:
                                    clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=kernel, taps=[t, 2], fv=[1 / (1 + b * taps_step), 1], fh=[1 / (1 + b * taps_step), 1])
                                elif chroma_direction == "h":
                                    clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=kernel, taps=[2, t], fv=[1, 1 / (1 + b * taps_step)], fh=[1, 1])
                                elif chroma_direction == "w":
                                    clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=kernel, taps=[2, t], fv=[1, 1], fh=[1, 1 / (1 + b * taps_step)])
                            if i:
                                clean_mangled = core.resize.Point(clean_mangled, transfer=1)
                            clean_mangled_16 = depth(clean_mangled, 16)
                            if clamp_bool:
                                clean_mangled_16 = core.akarin.Expr([clean_mangled_16], ["x 4096 60160 clamp", "x 4096 61440 clamp", "x 4096 61440 clamp"])
                            lowpassed_16 = depth(lowpassed_8_n, 16)
                            if plane == 1:
                                clean_mangled_16 = get_u(clean_mangled_16)
                                lowpassed_16 = get_u(lowpassed_16)
                            elif plane == 2:
                                clean_mangled_16 = get_v(clean_mangled_16)
                                lowpassed_16 = get_v(lowpassed_16)
                            else:
                                clean_mangled_16 = get_y(clean_mangled_16)
                                lowpassed_16 = get_y(lowpassed_16)
                            clean_mangled_16 = core.std.Crop(clean_mangled_16, 10, 10, 10, 10)
                            lowpassed_16 = core.std.Crop(lowpassed_16, 10, 10, 10, 10)
                            error_avg = core.std.PlaneStats(clean_mangled_16, lowpassed_16, 0, "PS")
                            error_n = error_avg.get_frame(0).props.PSDiff
                            error_info = [t, b, kernel, j, i, clamp_bool]
                            error_list.append([error_info, [error_n]])
                            if error_n < lowest_error:
                                error_string = f"{error_info[0]} taps, 1 / {round(1 + error_info[1] * taps_step, 3)} blur, {error_info[2]} kernel"
                                if report_bits:
                                    error_string += f", bit depth {error_info[3]}"
                                if report_ll:
                                    error_string += f", linear light as {error_info[4]}"
                                if report_clamp:
                                    error_string += f", clamp = {error_info[5]}"
                                lowest_error = error_n
                                print(f"Lowest error is now {error_string}")
    error_list.sort(key=lambda x: x[1][0])
    top_error_amount = len(bits) * 50
    top_error_dict = error_list[:top_error_amount]
    if len(frame_numbers) > 1:
        for n in frame_numbers:
            print(f"Now on frame {n}")
            frame_number = n
            clean_8_n = clean_8[frame_number]
            lowpassed_8_n = lowpassed_8[frame_number]
            clean_16 = depth(clean_8_n, 16)
            for i in top_error_dict:
                info_list = i[0]
                i_taps = info_list[0]
                i_blur = info_list[1]
                i_kernel = info_list[2]
                i_bits = info_list[3]
                i_ll = info_list[4]
                i_clamp = info_list[5]
                if i_bits == 8:
                    clean_1 = clean_8_n
                else:
                    clean_1 = clean_16
                if i_ll:
                    if i_bits == 8:
                        clean_2 = core.resize.Point(core.resize.Lanczos(clean_1, format=vs.YUV444P8), transfer=8)
                    else:
                        clean_2 = core.resize.Point(core.resize.Lanczos(clean_1, format=vs.YUV444P16), transfer=8)
                else:
                    clean_2 = clean_1
                if i_kernel == "bilinear":
                    if plane == 0:
                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=i_kernel, taps=[1, 2], fv=[1 / (1 + i_blur * taps_step), 1], fh=[1 / (1 + i_blur * taps_step), 1])
                    elif chroma_direction == "h":
                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=i_kernel, taps=[2, 1], fv=[1, 1 / (1 + i_blur * taps_step)], fh=[1, 1])
                    elif chroma_direction == "w":
                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=i_kernel, taps=[2, 1], fv=[1, 1], fh=[1, 1 / (1 + i_blur * taps_step)])
                elif i_kernel=="welch":
                    clean_mangled = Welch(blur=(1 + i_blur * taps_step), taps=i_taps).scale(clean_2, force=True)
                elif i_kernel=="bohman":
                    clean_mangled = Bohman(blur=(1 + i_blur * taps_step), taps=i_taps).scale(clean_2, force=True)
                else:
                    if plane == 0:
                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=i_kernel, taps=[i_taps, 2], fv=[1 / (1 + i_blur * taps_step), 1], fh=[1 / (1 + i_blur * taps_step), 1])
                    elif chroma_direction == "h":
                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=i_kernel, taps=[2, i_taps], fv=[1, 1 / (1 + i_blur * taps_step)], fh=[1, 1])
                    elif chroma_direction == "w":
                        clean_mangled = core.fmtc.resample(clean_2, h=clean_2.height, w=clean_2.width, kernel=i_kernel, taps=[2, i_taps], fv=[1, 1], fh=[1, 1 / (1 + i_blur * taps_step)])
                if i_ll:
                    clean_mangled = core.resize.Point(clean_mangled, transfer=1)
                clean_mangled_16 = depth(clean_mangled, 16)
                if i_clamp:
                    clean_mangled_16 = core.akarin.Expr([clean_mangled_16], ["x 4096 60160 clamp", "x 4096 61440 clamp", "x 4096 61440 clamp"])
                lowpassed_16 = depth(lowpassed_8_n, 16)
                if plane == 1:
                    clean_mangled_16 = get_u(clean_mangled_16)
                    lowpassed_16 = get_u(lowpassed_16)
                elif plane == 2:
                    clean_mangled_16 = get_v(clean_mangled_16)
                    lowpassed_16 = get_v(lowpassed_16)
                else:
                    clean_mangled_16 = get_y(clean_mangled_16)
                    lowpassed_16 = get_y(lowpassed_16)
                clean_mangled_16 = core.std.Crop(clean_mangled_16, 10, 10, 10, 10)
                lowpassed_16 = core.std.Crop(lowpassed_16, 10, 10, 10, 10)
                error_avg = core.std.PlaneStats(clean_mangled_16, lowpassed_16, 0, "PS")
                error_n = error_avg.get_frame(0).props.PSDiff
                i[1].append(error_n)
    final_list = []
    for i in top_error_dict:
        final_list.append([i[0], sum(i[1])])
    final_list.sort(key=itemgetter(1))
    error_thr = final_list[0][1]
    is_within_error_thr = True
    print("Best kernel first:")
    for i in final_list:
        info_list = i[0]
        taps = info_list[0]
        blur = info_list[1]
        kernel = info_list[2]
        bits = info_list[3]
        ll = info_list[4]
        clamp = info_list[5]
        if plane == 0:
            prefix = "Luma: "
        elif chroma_direction == "h":
            prefix = "Chroma height: "
        elif chroma_direction == "w":
            prefix = "Chroma width: "
        output_string = f"{prefix}{taps} taps, {round(1 + blur * taps_step, 3)} blur, {kernel} kernel"
        if report_bits:
            output_string += f", bit depth {bits}"
        if report_ll:
            output_string += f", linear light as {ll}"
        if report_clamp:
            output_string += f", clamp = {clamp}"
        print(output_string)
        if i[1] > error_thr * 1.01:
            break

# if os.path.exists("clean.png"):
#     clean_8 = initialize_clip(core.bs.VideoSource("clean.png"), 8, 0, 13, 1).resize.Lanczos(format=vs.YUV420P8, matrix=1, transfer=1, primaries=1)
# else:
#     clean_8 = initialize_clip(core.bs.VideoSource("clean.webp"), 8, 0, 13, 1).resize.Lanczos(format=vs.YUV420P8, matrix=1, transfer=1, primaries=1)
# if os.path.exists("lowpassed.png"):
#     lowpassed_8 = initialize_clip(core.bs.VideoSource("lowpassed.png"), 8, 0, 13, 1).resize.Lanczos(format=vs.YUV420P8, matrix=1, transfer=1, primaries=1)
# else:
#     lowpassed_8 = initialize_clip(core.bs.VideoSource("lowpassed.webp"), 8, 0, 13, 1).resize.Lanczos(format=vs.YUV420P8, matrix=1, transfer=1, primaries=1)
# frame_numbers = [0] #if you're testing images, obviously you can only pick 1 frame

clean_8 = initialize_clip(core.bs.VideoSource(R"C:\Users\sing4\Downloads\Houseki no Kuni (English Dub) [HIDIVE VRV 1080p]\[babon-rip] Houseki no Kuni - 02 (English Dub) [HIDIVE VRV 1080p] [EA7F2A05].mkv"), 8, 1, 1, 1)[24:]
lowpassed_8 = initialize_clip(core.bs.VideoSource(R"C:\Users\sing4\Downloads\[BDMV][アニメ][171220] 宝石の国 Vol.1\LAND_OF_THE_LUSTROUS_1\BDMV\STREAM\00001.m2ts"), 8, 1, 1, 1)
frame_numbers = [32578, 32674] #testing multiple frames is recommended

clamp = [False, True] #see the delowpass guide for information about what to set this to. Or you can leave it like this and test both! (Normally there will be no difference.)
ll = [False] #add "True" if you want to test ll (benefit unclear)
bits = [16] #add "8" if you also want to test 8-bit (benefit unclear)
kernels = ["bilinear", "lanczos", "blackman", "blackmanminlobe", "welch", "bohman"]
plane = 0 #You should probably be testing different frames for luma and chroma, so this script does not support testing them all at once
chroma_direction = "h" #options are h and w. Chroma handling is often different between height and width.

upper_taps_limit = 12 #increase this if you are getting ~12 taps as the best result, obviously
upper_blur_limit = 1.5

if is_preview():
    if plane == 0:
        set_output(get_y(clean_8))
        set_output(get_y(lowpassed_8))
    if plane == 1:
        set_output(get_u(clean_8))
        set_output(get_u(lowpassed_8))
    if plane == 2:
        set_output(get_v(clean_8))
        set_output(get_v(lowpassed_8))
else:
    delowpass_bruteforcer(frame_numbers, clean_8, lowpassed_8, ll, bits, kernels, plane, clamp, chroma_direction, upper_taps_limit, upper_blur_limit)