import earnings_growth as eg
import time
import pandas as pd
import alpha_tools.utilities.utilities as utils

if __name__ == "__main__":
    # --------------------------------
    # params

    # --------------------------------
    # common data
    common_data_path = r"\\nas03-chuorui\Factors_intern\huqiang\data"

    unv_name = "universe4"
    unv = pd.read_csv(f"{common_data_path}/{unv_name}_1d.csv", index_col=0)
    unv.index = pd.to_datetime(unv.index.astype(str))

    industry_df = pd.read_csv(f"{common_data_path}/industry/industry_1d.csv", index_col=0)
    industry_df.index = pd.to_datetime(industry_df.index.astype(str), errors='coerce')

    # --------------------------------
    # data
    data_path = r"\\nas03-chuorui\Factors_intern\huqiang\factors\raw\task7"
    data_path1 = r"\\nas03-chuorui\Factors_intern\huqiang\data\处理后数据"
    data_path2 = r"\\nas03-chuorui\Factors_intern\huqiang\data\original"

    # df_op = pd.read_csv(f"{data_path1}/净利润股东权益.csv")
    # df_op['trade_date'] = pd.to_datetime(df_op['trade_date'].astype(str))

    # df_op = pd.read_csv(f"{data_path}/earing_growth_1season_winsor0_disc.csv",index_col=0)
    # df_op.index = pd.to_datetime(df_op.index.astype(str))

    # df_op = pd.read_csv(f"{data_path}/earing_growth_1y_winsor0_disc.csv",index_col=0)
    # df_op.index = pd.to_datetime(df_op.index.astype(str))

    # df_op = pd.read_csv(f"{data_path}/earingdiff1s_div_bet_winsor0_disc.csv", index_col=0)
    # df_op.index = pd.to_datetime(df_op.index.astype(str))

    df_op = pd.read_csv(f"{data_path}/earingdiff1y_div_bet_winsor0_disc.csv", index_col=0)
    df_op.index = pd.to_datetime(df_op.index.astype(str))


    # --------------------------------
    # factors
    tic = time.time()

    # f = eg.earing_growth(df_op, unv, if_winsor=True, winsor=0)
    # out_name = 'earing_growth_1season_winsor0_disc'
    #
    # f = eg.get_fill(df_op, universe=unv, if_universe=True)
    # out_name = 'earing_growth_1season_winsor0_ffill_unv'

    # f = eg.get_fill(df_op, universe=unv, if_universe=False)
    # out_name = 'earing_growth_1season_winsor0_ffill_rp'

    # f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=True)
    # out_name = 'earing_growth_1season_winsor0_decay90_unv'

    # f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=False)
    # out_name = 'earing_growth_1season_winsor0_decay90_rp'

    # f = eg.earing_growth1(df_op, unv, if_winsor=True, winsor=0)
    # out_name = 'earing_growth_1y_winsor0_disc'
    #
    # f = eg.get_fill(df_op, universe=unv, if_universe=True)
    # out_name = 'earing_growth_1y_winsor0_ffill_unv'

    # f = eg.get_fill(df_op, universe=unv, if_universe=False)
    # out_name = 'earing_growth_1y_winsor0_ffill_rp'

    # f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=True)
    # out_name = 'earing_growth_1y_winsor0_decay90_unv'

    # f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=False)
    # out_name = 'earing_growth_1y_winsor0_decay90_rp'

    # f = eg.earing_growth2(df_op, unv, if_winsor=True, winsor=0)
    # out_name = 'earingdiff1s_div_bet_winsor0_disc'

    # f = eg.get_fill(df_op, universe=unv, if_universe=True)
    # out_name = 'earingdiff1s_div_bet_winsor0_ffill_unv'

    # f = eg.get_fill(df_op, universe=unv, if_universe=False)
    # out_name = 'earingdiff1s_div_bet_winsor0_ffill_rp'

    # f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=True)
    # out_name = 'earingdiff1s_div_bet_winsor0_decay90_unv'

    # f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=False)
    # out_name = 'earingdiff1s_div_bet_winsor0_decay90_rp'

    # f = eg.earing_growth3(df_op, unv, if_winsor=True, winsor=0)
    # out_name = 'earingdiff1y_div_bet_winsor0_disc'

    # f = eg.get_fill(df_op, universe=unv, if_universe=True)
    # out_name = 'earingdiff1y_div_bet_winsor0_ffill_unv'

    # f = eg.get_fill(df_op, universe=unv, if_universe=False)
    # out_name = 'earingdiff1y_div_bet_winsor0_ffill_rp'

    # f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=True)
    # out_name = 'earingdiff1y_div_bet_winsor0_decay90_unv'

    f = eg.get_decay(df_op, universe=unv, decay=0.9, if_universe=False)
    out_name = 'earingdiff1y_div_bet_winsor0_decay90_rp'

    print(f"done, takes {round(time.time() - tic, 2)}s.")
    out_path = r"\\nas03-chuorui\Factors_intern\huqiang\factors\raw\task7"
    out_path1 = r"\\nas03-chuorui\Factors_intern\huqiang\data\处理后数据"
    utils.safely_to_csv(
        to_csv_object=f,
        path=out_path,
        name=out_name,
        index=True,
        header=True,
    )
