import numpy as np

from track import gen_track_from_file
from ground import Ground
from heuristic import gen_tc_sol, get_tc_variables_from_v_u_w, gen_warm_start_data
from optimize import VAO, TC, EETC_VAO
from train import Train


def get_vao():
    """get all VAO cases"""
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ###############################################################################
    ground_names: list[str] = [
        "gd1", "gd2", "gd3", "gd4", "gd5", "gd6",
        "gd_hp1_100", "gd_hp1_50",
        "gd_hp2_100", "gd_hp2_50",
        "gd_hp3_50",
        "gd_gaoyan"
    ]
    ###############################################################################
    for gn in ground_names:
        gd = Ground(name=gn)
        vao = VAO(ground=gd, VI_on=True, LC_ON=True, plot_ground=True)
        vao.optimize(save_on=True, TimeLimit=4 * 3600)

        trk = vao.get_track()


def get_sotc():
    """get all track from json files, then apply heuristics to get SOTC warm start, then optimize SOTC to gap zero."""
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##############################################################################
    ground_folder = r"Cases\vao_LC_VI"
    ground_names: list[str] = [
        "gd1", "gd2", "gd3", "gd4", "gd5", "gd6",
        "gd_hp1_100", "gd_hp1_50",
        "gd_hp2_100", "gd_hp2_50",
        "gd_hp3_50",
        "gd_gaoyan"
    ]
    train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    ###############################################################################

    for gn in ground_names:
        gd = Ground(name=gn)
        trk = gen_track_from_file(ground=gd, json_file=f"{ground_folder}\\{gn}____vao_TC_VI\\{gn}____vao_TC_VI.json")
        for tn in train_names:
            tr = Train(name=tn)
            vd_arr, cd_arr, wd_arr, vu_arr, cu_arr, wu_arr = gen_tc_sol(trk=trk, tr=tr, is_ee=False)
            wsd = {"tc_d": get_tc_variables_from_v_u_w(vd_arr, cd_arr, wd_arr, tr, trk, is_downhill_dir=True),
                   "tc_u": get_tc_variables_from_v_u_w(vu_arr, cu_arr, wu_arr, tr, trk, is_downhill_dir=False)}

            sotc = TC(train=tr, track=trk, is_ee=False, warm_start_data=wsd, debug_mode=False)
            sotc.optimize(save_on=True)
    return


def get_eetc():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##############################################################################
    ground_folder = r"Cases\vao_LC_VI"
    ground_names: list[str] = [
        # "gd1", "gd2", "gd3", "gd4", "gd5", "gd6",
        # "gd_hp1_100", "gd_hp1_50",
        # "gd2", "gd4",
        # "gd_hp2_100", "gd_hp2_50",
        # "gd_hp3_50",
        "gd_gaoyan"
    ]
    train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    ###############################################################################

    for gn in ground_names:
        gd = Ground(name=gn)
        trk = gen_track_from_file(ground=gd, json_file=f"{ground_folder}\\{gn}____vao_LC_VI\\{gn}____vao_LC_VI.json")
        for tn in train_names:
            tr = Train(name=tn)
            # use sotc as a warm start
            vd_arr, cd_arr, wd_arr, vu_arr, cu_arr, wu_arr = gen_tc_sol(trk=trk, tr=tr, is_ee=True, display_on=True)
            wsd = {"tc_d": get_tc_variables_from_v_u_w(vd_arr, cd_arr, wd_arr, tr, trk, is_downhill_dir=True),
                   "tc_u": get_tc_variables_from_v_u_w(vu_arr, cu_arr, wu_arr, tr, trk, is_downhill_dir=False)}

            eetc = TC(train=tr, track=trk, is_ee=True, warm_start_data=wsd, debug_mode=False)
            eetc.optimize(save_on=True, MIPGap=0.0001)
    return


def get_eetc_vao_with_LC():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##################################################################################
    ground_names: list[str] = [
        # "gd2", "gd4",
        # "gd_hp2_100", "gd_hp2_50",
        "gd_hp3_50",
        "gd_gaoyan"
    ]
    train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    ##################################################################################
    for gn in ground_names:
        gd = Ground(name=gn)
        for tn in train_names:
            tr = Train(name=tn)
            ev = EETC_VAO(ground=gd, train=tr, tcVI_on=False, VI_on=False, LC_on=True)
            ev.optimize(TimeLimit=3600 * 24)
    return


def get_eetc_vao_with_VI():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##################################################################################
    ground_names: list[str] = [
        # "gd2", "gd4",
        # "gd_hp2_100", "gd_hp2_50",
        "gd_hp3_50",
        "gd_gaoyan"
    ]
    train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    ##################################################################################
    for gn in ground_names:
        gd = Ground(name=gn)
        for tn in train_names:
            tr = Train(name=tn)
            ev = EETC_VAO(ground=gd, train=tr, tcVI_on=False, VI_on=True, LC_on=False)
            ev.optimize(TimeLimit=3600 * 24)
    return


def get_eetc_vao_with_tcVI():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##################################################################################
    ground_names: list[str] = [
        # "gd2", "gd4",
        # "gd_hp2_100", "gd_hp2_50",
        "gd_hp3_50",
        "gd_gaoyan"
    ]
    train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    ##################################################################################
    for gn in ground_names:
        gd = Ground(name=gn)
        for tn in train_names:
            tr = Train(name=tn)
            ev = EETC_VAO(ground=gd, train=tr, tcVI_on=True, VI_on=False, LC_on=False)
            ev.optimize(TimeLimit=3600 * 24)
    return


def get_eetc_vao_with_WS():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##################################################################################
    ground_names: list[str] = [
        # "gd2", "gd4",
        # "gd_hp2_100", "gd_hp2_50",
        "gd_hp3_50",
        "gd_gaoyan"
    ]
    train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    ##################################################################################
    for gn in ground_names:
        gd = Ground(name=gn)
        trk = gen_track_from_file(ground=gd, json_file=f"Cases\\vao_LC_VI\\{gn}____vao_TC_VI\\{gn}____vao_TC_VI.json")
        for tn in train_names:
            tr = Train(name=tn)
            wsd = gen_warm_start_data(trk=trk, tr=tr, is_ee=True)
            ev = EETC_VAO(ground=gd, train=tr, tcVI_on=False, VI_on=False, LC_on=False, warm_start_data=wsd)
            ev.optimize(TimeLimit=3600 * 24)
    return


def main():
    # get_eetc()
    pass


if __name__ == '__main__':
    main()
