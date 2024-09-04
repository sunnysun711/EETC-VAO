from ground import Ground
from heuristic import gen_track, gen_warm_start_data
from optimize import EETC_VAO, VAO, TC
from track import gen_track_from_file
from train import Train


def get_eetc_vao():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##############################################################################
    ground_names: list[str] = [
        # "gd1", "gd2", "gd3", "gd4", "gd5", "gd6",
        # "gd_hp1_100", "gd_hp1_50",
        "gd2", "gd4",
        "gd_hp2_100", "gd_hp2_50",
        "gd_hp3_50",
        "gd_gaoyan"
    ]
    train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    ###############################################################################
    for gn in ground_names:
        gd = Ground(name=gn)
        for tn in train_names:
            tr = Train(name=tn)
            # benchmark
            ev = EETC_VAO(ground=gd, train=tr, tcVI_on=False, VI_on=False, LC_on=False)
            ev.optimize(TimeLimit=3600 * 24)

    pass


def get_eetc_vao_with_cuts():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##################################################################################
    ground_names: list[str] = [
        # "gd2", "gd4",
        # "gd_hp2_100", "gd_hp2_50",
        # "gd_hp3_50",
        "gd_gaoyan"
    ]
    # train_names: list[str] = ["CRH380AL", "HXD1D", "HXD2"]
    train_names: list[str] = ["HXD2"]
    ##################################################################################
    for gn in ground_names:
        gd = Ground(name=gn)
        for tn in train_names:
            tr = Train(name=tn)
            ev = EETC_VAO(ground=gd, train=tr, tcVI_on=True, VI_on=True, LC_on=True)
            ev.optimize(TimeLimit=3600 * 24)
    return


def get_eetc_vao_with_warm_start():
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


def get_eetc_vao_with_cuts_and_warm_start():
    import matplotlib
    matplotlib.use("Agg")  # 使用 Agg 后端，适用于生成图片但不需要显示
    ##################################################################################
    ground_names: list[str] = [
        "gd2", "gd4",
        "gd_hp2_100", "gd_hp2_50",
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
            ev = EETC_VAO(ground=gd, train=tr, tcVI_on=True, VI_on=True, LC_on=True, warm_start_data=wsd)
            ev.optimize(TimeLimit=3600 * 24)
    return


def main():
    # get_eetc_vao()
    get_eetc_vao_with_cuts()
    # get_eetc_vao_with_cuts_and_warm_start()
    pass


if __name__ == '__main__':
    main()
