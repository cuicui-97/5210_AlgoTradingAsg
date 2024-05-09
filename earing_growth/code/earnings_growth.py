import numpy as np
import pandas as pd
import alpha_tools.utilities.utilities as utils
import alpha_tools.utilities.operators as operators
from joblib import Parallel, delayed


# --------------------------------------------------------------------------------
# 功能函数

def get_indzscore(df_op, industry_df, universe=None):
    """
    对df_op进行indzscore

        Parameters
        ----------
         df_op: pd.DataFrame
           df_op: index=Timestamp, columns=code
        industry_df: pd.DataFrame
        universe: Union[pd.DataFrame, None]
            default None, universe for volume

        Returns
        -------
        f: pd.DataFrame
            index=Timestamp, columns=code
    """

    ind_codes = list(industry_df.replace([np.inf, -np.inf], 0).stack().unique())
    ind_codes.remove(0)
    f_lst = []

    for ind_code in ind_codes:
        if universe is not None:
            df_ind = utils.get_universe(alpha=df_op,
                                        universe=universe,
                                        industry_df=industry_df,
                                        industry_code=ind_code)
        else:
            industry_unv = pd.DataFrame(np.where(industry_df == ind_code, 1, np.nan),
                                        index=industry_df.index,
                                        columns=industry_df.columns)

            df_ind = utils.get_universe(alpha=df_op,
                                        universe=industry_unv,
                                        industry_df=None,
                                        industry_code=-1)

        df_ind = df_ind.dropna(how="all", axis=1)
        # ind_zscore
        df_ind = operators.cx_scale(alpha=df_ind,
                                    method="z-score",
                                    fill_illegal="ffill",
                                    illegal_tol=1e-10)
        f1 = df_ind.dropna(how="all")
        f_lst.append(f1)

    f1 = pd.concat(f_lst, axis=1).dropna(how="all")
    f1 = f1.groupby(f1.columns, axis=1).first()
    f1 = f1.replace([np.inf, -np.inf], np.nan)
    f1 = f1.T.sort_index().T
    f1 = f1.sort_index()
    # f1 = f1.reindex(index=universe.index, columns=universe.columns).ffill()
    f1.index.name = 'trade_date'
    return f1


def get_fill(df_op, universe, if_universe=True):
    """
        对df_op进行fill

        Parameters
        ----------
        df_op: pd.DataFrame
            df_op:index=Timestamp, columns=code
        universe: Union[pd.DataFrame, None]
            default None, universe for volume
        if_universe:bool
            default True, 是否对非交易日期间也进行ffill
        Returns
        -------
        f: pd.DataFrame
                index=Timestamp, columns=code
    """
    if if_universe:
        f = df_op.reindex(index=universe.index).ffill()
    else:
        f = df_op.ffill()
    f = f.replace([np.inf, -np.inf], np.nan)
    f.dropna(how='all', inplace=True)
    f.index.name = 'trade_date'

    return f


def get_decay(df_op, universe, decay, if_universe=True):
    """
        对df_op进行decay
        ----------
        df_op: pd.DataFrame
            df_op:index=Timestamp, columns=code
        decay:float
            衰减因子
        universe: Union[pd.DataFrame, None]
            default None, universe for volume
        if_universe:bool
            default True, 是否对非交易日期间也进行ffill

        Returns
        -------
        f: pd.DataFrame
                index=Timestamp, columns=code
    """
    f = df_op.reindex(index=universe.index)
    for i in range(1, len(f.index)):
        f.loc[f.index[i]] = f.loc[f.index[i]].fillna(f.iloc[i - 1] * decay)
    if not if_universe:
        f = f.reindex(index=df_op.index)

    f = f.replace([np.inf, -np.inf], np.nan)
    f.dropna(how='all', inplace=True)
    f.index.name = 'trade_date'

    return f


# --------------------------------------------------------------------------------
def func1(df0, stock, trade_dt_field, report_peirod_field, op_field1, op_field2, if_winsor, winsor):
    """
    获得Earning_growth数据
    （Earningst+r − Earningst）/Bet，r=1s
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)
    """
    dictmonth = {3: 4, 6: 8, 9: 10, 12: 4}
    op1_lst = []
    op1_last_lst = []
    op2_last_lst = []

    td_lst = []
    df0 = df0.reset_index()
    trade_dates = sorted(df0[trade_dt_field].unique())
    trade_dates = pd.to_datetime(trade_dates)
    for trade_dt in trade_dates:
        # history data
        df_now = df0[df0[trade_dt_field] <= trade_dt]
        # new data
        newest_report_date = df_now[report_peirod_field].max()

        if if_winsor:
            month = newest_report_date.month
            year = newest_report_date.year
            if month == 12:
                if trade_dt.month > (dictmonth[month] + winsor) or trade_dt.year > year + 1:
                    continue
            else:
                if trade_dt.month > dictmonth[month] or trade_dt.year > year:
                    continue

        idx_this = df_now[df_now[report_peirod_field] == newest_report_date].index
        op = df_now.loc[idx_this[-1], op_field1]
        op1_lst.append(op)
        td_lst.append(trade_dt)
        
        pre_report_date = newest_report_date - pd.offsets.QuarterEnd()
        idx_last = df_now[df_now[report_peirod_field] == pre_report_date].index
        if not len(idx_last):
            op1_last_lst.append(np.nan)
            op2_last_lst.append(np.nan)
            continue
        op1_last = df_now.loc[idx_last[-1], op_field1]
        op1_last_lst.append(op1_last)
        op2_last = df_now.loc[idx_last[-1], op_field2]
        op2_last_lst.append(op2_last)
    op1_values = pd.Series(data=op1_lst, index=td_lst)
    op1_last_values = pd.Series(data=op1_last_lst, index=td_lst)
    op2_last_values = pd.Series(data=op2_last_lst, index=td_lst)
    diff = op1_values.sub(op1_last_values)
    f = diff.div(op2_last_values)
    f.name = stock
    return f


def func11(df0, stock, trade_dt_field, report_peirod_field, op_field1, op_field2, if_winsor, winsor):
    """
    获得Earning_growth数据
    （Earningst+r − Earningst）/Bet，r=1y
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)
    """
    dictmonth = {3: 4, 6: 8, 9: 10, 12: 4}
    op1_lst = []
    op1_last_lst = []
    op2_last_lst = []

    td_lst = []
    df0 = df0.reset_index()
    trade_dates = sorted(df0[trade_dt_field].unique())
    trade_dates = pd.to_datetime(trade_dates)
    for trade_dt in trade_dates:
        # history data
        df_now = df0[df0[trade_dt_field] <= trade_dt]
        # new data
        newest_report_date = df_now[report_peirod_field].max()

        if if_winsor:
            month = newest_report_date.month
            year = newest_report_date.year
            if month == 12:
                if trade_dt.month > (dictmonth[month] + winsor) or trade_dt.year > year + 1:
                    continue
            else:
                if trade_dt.month > dictmonth[month] or trade_dt.year > year:
                    continue

        idx_this = df_now[df_now[report_peirod_field] == newest_report_date].index
        op = df_now.loc[idx_this[-1], op_field1]
        op1_lst.append(op)
        td_lst.append(trade_dt)

        pre_report_date = newest_report_date - pd.DateOffset(years=1)
        idx_last = df_now[df_now[report_peirod_field] == pre_report_date].index
        if not len(idx_last):
            op1_last_lst.append(np.nan)
            op2_last_lst.append(np.nan)
            continue
        op1_last = df_now.loc[idx_last[-1], op_field1]
        op1_last_lst.append(op1_last)
        op2_last = df_now.loc[idx_last[-1], op_field2]
        op2_last_lst.append(op2_last)
    op1_values = pd.Series(data=op1_lst, index=td_lst)
    op1_last_values = pd.Series(data=op1_last_lst, index=td_lst)
    op2_last_values = pd.Series(data=op2_last_lst, index=td_lst)
    diff = op1_values.sub(op1_last_values)
    f = diff.div(op2_last_values)
    f.name = stock
    return f


def earing_growth(df_op, universe, if_winsor, winsor):
    """
    （Earningst+r − Earningst）/Bet，r=1s
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)

    Parameters
    ----------
    df_op: pd.DataFrame
    universe: Union[pd.DataFrame, None]
        default None, universe for volume
    if_winsor:bool
        default True:是否对未在规定时间公告的股票进行剔除
    winsor:int
       default 0:剔除的时间范围

    Returns
    -------
    f: pd.DataFrame
        index=Timestamp, columns=code
    """
    df_op = df_op.sort_values(by=['trade_date', 'REPORT_PERIOD']).reset_index()
    df_op['REPORT_PERIOD'] = pd.to_datetime(df_op['REPORT_PERIOD'].astype(str), errors='coerce')
    df_op = df_op[~df_op["S_INFO_WINDCODE"].str.match(r"^[a-zA-Z]")]
    df_op = df_op.drop_duplicates(subset=['S_INFO_WINDCODE', 'trade_date', 'REPORT_PERIOD',
                                          'TOT_SHRHLDR_EQY_EXCL_MIN_INT', 'NET_PROFIT_EXCL_MIN_INT_INC'], keep='first')

    def process_stock_parallel(stock):
        df = df_op[df_op['S_INFO_WINDCODE'] == stock]
        ratio = func1(df0=df,
                      stock=stock,
                      trade_dt_field='trade_date',
                      report_peirod_field='REPORT_PERIOD',
                      op_field1='NET_PROFIT_EXCL_MIN_INT_INC',
                      op_field2='TOT_SHRHLDR_EQY_EXCL_MIN_INT',
                      if_winsor=if_winsor,
                      winsor=winsor)
        return ratio

    results = Parallel(n_jobs=4)(delayed(process_stock_parallel)(stock) for stock in df_op['S_INFO_WINDCODE'].unique())
    df_ration = pd.concat(results, axis=1, join='outer')
    df_ration.index = pd.to_datetime(df_ration.index.astype(str))
    df_ration.index.name = 'trade_date'
    ration_list = sorted(df_ration.index)
    unv_list = sorted(universe.index)
    union_list = list(set(ration_list).union(set(unv_list)))
    union_list = sorted(union_list)
    f = df_ration.reindex(index=union_list)
    for i in union_list:
        if i in unv_list:
            continue
        index = union_list.index(i)
        for j in union_list[index + 1:]:
            if j not in unv_list:
                continue
            f.loc[j] = f.loc[j].fillna(f.loc[i])
            break
    f = f.reindex(index=universe.index)
    f = f.dropna(how='all')
    return f


def earing_growth1(df_op, universe, if_winsor, winsor):
    """
    （Earningst+r − Earningst）/Bet，r=1y
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)
    用的是上一季度的数据
    Parameters
    ----------
    df_op: pd.DataFrame
    universe: Union[pd.DataFrame, None]
        default None, universe for volume
    if_winsor:bool
        default True:是否对未在规定时间公告的股票进行剔除
    winsor:int
       default 0:剔除的时间范围

    Returns
    -------
    f: pd.DataFrame
        index=Timestamp, columns=code
    """
    df_op = df_op.sort_values(by=['trade_date', 'REPORT_PERIOD']).reset_index()
    df_op['REPORT_PERIOD'] = pd.to_datetime(df_op['REPORT_PERIOD'].astype(str), errors='coerce')
    df_op = df_op[~df_op["S_INFO_WINDCODE"].str.match(r"^[a-zA-Z]")]
    df_op = df_op.drop_duplicates(subset=['S_INFO_WINDCODE', 'trade_date', 'REPORT_PERIOD',
                                          'TOT_SHRHLDR_EQY_EXCL_MIN_INT', 'NET_PROFIT_EXCL_MIN_INT_INC'], keep='first')

    def process_stock_parallel(stock):
        df = df_op[df_op['S_INFO_WINDCODE'] == stock]
        ratio = func11(df0=df,
                       stock=stock,
                       trade_dt_field='trade_date',
                       report_peirod_field='REPORT_PERIOD',
                       op_field1='NET_PROFIT_EXCL_MIN_INT_INC',
                       op_field2='TOT_SHRHLDR_EQY_EXCL_MIN_INT',
                       if_winsor=if_winsor,
                       winsor=winsor)
        return ratio

    results = Parallel(n_jobs=4)(delayed(process_stock_parallel)(stock) for stock in df_op['S_INFO_WINDCODE'].unique())
    df_ration = pd.concat(results, axis=1, join='outer')
    df_ration.index = pd.to_datetime(df_ration.index.astype(str))
    df_ration.index.name = 'trade_date'
    ration_list = sorted(df_ration.index)
    unv_list = sorted(universe.index)
    union_list = list(set(ration_list).union(set(unv_list)))
    union_list = sorted(union_list)
    f = df_ration.reindex(index=union_list)
    for i in union_list:
        if i in unv_list:
            continue
        index = union_list.index(i)
        for j in union_list[index + 1:]:
            if j not in unv_list:
                continue
            f.loc[j] = f.loc[j].fillna(f.loc[i])
            break
    f = f.reindex(index=universe.index)
    f = f.dropna(how='all')
    return f


# --------------------------------------------------------------------------------
def func2(df0, stock, trade_dt_field, report_peirod_field, op_field1, op_field2, if_winsor, winsor):
    """
    获得Earning_growth数据
    （Earningst+r − Earningst）/Bet+r，r=1s
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)
    """
    dictmonth = {3: 4, 6: 8, 9: 10, 12: 4}
    op1_lst = []
    op1_last_lst = []
    op2_lst = []

    td_lst = []
    df0 = df0.reset_index()
    trade_dates = sorted(df0[trade_dt_field].unique())
    trade_dates = pd.to_datetime(trade_dates)
    for trade_dt in trade_dates:
        # history data
        df_now = df0[df0[trade_dt_field] <= trade_dt]
        # new data
        newest_report_date = df_now[report_peirod_field].max()

        if if_winsor:
            month = newest_report_date.month
            year = newest_report_date.year
            if month == 12:
                if trade_dt.month > (dictmonth[month] + winsor) or trade_dt.year > year + 1:
                    continue
            else:
                if trade_dt.month > dictmonth[month] or trade_dt.year > year:
                    continue

        idx_this = df_now[df_now[report_peirod_field] == newest_report_date].index
        op = df_now.loc[idx_this[-1], op_field1]
        op1_lst.append(op)
        op2 = df_now.loc[idx_this[-1], op_field2]
        op2_lst.append(op2)
        td_lst.append(trade_dt)

        pre_report_date = newest_report_date - pd.offsets.QuarterEnd()
        idx_last = df_now[df_now[report_peirod_field] == pre_report_date].index
        if not len(idx_last):
            op1_last_lst.append(np.nan)
            continue
        op1_last = df_now.loc[idx_last[-1], op_field1]
        op1_last_lst.append(op1_last)

    op1_values = pd.Series(data=op1_lst, index=td_lst)
    op1_last_values = pd.Series(data=op1_last_lst, index=td_lst)
    op2_values = pd.Series(data=op2_lst, index=td_lst)
    diff = op1_values.sub(op1_last_values)
    f = diff.div(op2_values)
    f.name = stock
    return f


def func21(df0, stock, trade_dt_field, report_peirod_field, op_field1, op_field2, if_winsor, winsor):
    """
    获得Earning_growth数据
    （Earningst+r − Earningst）/Bet+r，r=1y
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)
    """
    dictmonth = {3: 4, 6: 8, 9: 10, 12: 4}
    op1_lst = []
    op1_last_lst = []
    op2_lst = []

    td_lst = []
    df0 = df0.reset_index()
    trade_dates = sorted(df0[trade_dt_field].unique())
    trade_dates = pd.to_datetime(trade_dates)
    for trade_dt in trade_dates:
        # history data
        df_now = df0[df0[trade_dt_field] <= trade_dt]
        # new data
        newest_report_date = df_now[report_peirod_field].max()

        if if_winsor:
            month = newest_report_date.month
            year = newest_report_date.year
            if month == 12:
                if trade_dt.month > (dictmonth[month] + winsor) or trade_dt.year > year + 1:
                    continue
            else:
                if trade_dt.month > dictmonth[month] or trade_dt.year > year:
                    continue

        idx_this = df_now[df_now[report_peirod_field] == newest_report_date].index
        op = df_now.loc[idx_this[-1], op_field1]
        op1_lst.append(op)
        op2 = df_now.loc[idx_this[-1], op_field2]
        op2_lst.append(op2)
        td_lst.append(trade_dt)

        pre_report_date = newest_report_date - pd.DateOffset(years=1)
        idx_last = df_now[df_now[report_peirod_field] == pre_report_date].index
        if not len(idx_last):
            op1_last_lst.append(np.nan)
            continue
        op1_last = df_now.loc[idx_last[-1], op_field1]
        op1_last_lst.append(op1_last)

    op1_values = pd.Series(data=op1_lst, index=td_lst)
    op1_last_values = pd.Series(data=op1_last_lst, index=td_lst)
    op2_values = pd.Series(data=op2_lst, index=td_lst)
    diff = op1_values.sub(op1_last_values)
    f = diff.div(op2_values)
    f.name = stock
    return f


def earing_growth2(df_op, universe, if_winsor, winsor):
    """
    （Earningst+r − Earningst）/Bet+r，R=1S
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)

    Parameters
    ----------
    df_op: pd.DataFrame
    universe: Union[pd.DataFrame, None]
        default None, universe for volume
    if_winsor:bool
        default True:是否对未在规定时间公告的股票进行剔除
    winsor:int
       default 0:剔除的时间范围

    Returns
    -------
    f: pd.DataFrame
        index=Timestamp, columns=code
    """
    df_op = df_op.sort_values(by=['trade_date', 'REPORT_PERIOD']).reset_index()
    df_op['REPORT_PERIOD'] = pd.to_datetime(df_op['REPORT_PERIOD'].astype(str), errors='coerce')
    df_op = df_op[~df_op["S_INFO_WINDCODE"].str.match(r"^[a-zA-Z]")]
    df_op = df_op.drop_duplicates(subset=['S_INFO_WINDCODE', 'trade_date', 'REPORT_PERIOD',
                                          'TOT_SHRHLDR_EQY_EXCL_MIN_INT', 'NET_PROFIT_EXCL_MIN_INT_INC'], keep='first')

    def process_stock_parallel(stock):
        df = df_op[df_op['S_INFO_WINDCODE'] == stock]
        ratio = func2(df0=df,
                      stock=stock,
                      trade_dt_field='trade_date',
                      report_peirod_field='REPORT_PERIOD',
                      op_field1='NET_PROFIT_EXCL_MIN_INT_INC',
                      op_field2='TOT_SHRHLDR_EQY_EXCL_MIN_INT',
                      if_winsor=if_winsor,
                      winsor=winsor)
        return ratio

    results = Parallel(n_jobs=4)(delayed(process_stock_parallel)(stock) for stock in df_op['S_INFO_WINDCODE'].unique())
    df_ration = pd.concat(results, axis=1, join='outer')
    df_ration.index = pd.to_datetime(df_ration.index.astype(str))
    df_ration.index.name = 'trade_date'
    ration_list = sorted(df_ration.index)
    unv_list = sorted(universe.index)
    union_list = list(set(ration_list).union(set(unv_list)))
    union_list = sorted(union_list)
    f = df_ration.reindex(index=union_list)
    for i in union_list:
        if i in unv_list:
            continue
        index = union_list.index(i)
        for j in union_list[index + 1:]:
            if j not in unv_list:
                continue
            f.loc[j] = f.loc[j].fillna(f.loc[i])
            break
    f = f.reindex(index=universe.index)
    f = f.dropna(how='all')
    return f


def earing_growth3(df_op, universe, if_winsor, winsor):
    """
    （Earningst+r − Earningst）/Bet+r，R=1y
    Earnings growth is calculated as future earnings minus current earnings, scaled by current book equity (BE)
    Parameters
    ----------
    df_op: pd.DataFrame
    universe: Union[pd.DataFrame, None]
        default None, universe for volume
    if_winsor:bool
        default True:是否对未在规定时间公告的股票进行剔除
    winsor:int
       default 0:剔除的时间范围

    Returns
    -------
    f: pd.DataFrame
        index=Timestamp, columns=code
    """
    df_op = df_op.sort_values(by=['trade_date', 'REPORT_PERIOD']).reset_index()
    df_op['REPORT_PERIOD'] = pd.to_datetime(df_op['REPORT_PERIOD'].astype(str), errors='coerce')
    df_op = df_op[~df_op["S_INFO_WINDCODE"].str.match(r"^[a-zA-Z]")]
    df_op = df_op.drop_duplicates(subset=['S_INFO_WINDCODE', 'trade_date', 'REPORT_PERIOD',
                                          'TOT_SHRHLDR_EQY_EXCL_MIN_INT', 'NET_PROFIT_EXCL_MIN_INT_INC'], keep='first')

    def process_stock_parallel(stock):
        df = df_op[df_op['S_INFO_WINDCODE'] == stock]
        ratio = func21(df0=df,
                       stock=stock,
                       trade_dt_field='trade_date',
                       report_peirod_field='REPORT_PERIOD',
                       op_field1='NET_PROFIT_EXCL_MIN_INT_INC',
                       op_field2='TOT_SHRHLDR_EQY_EXCL_MIN_INT',
                       if_winsor=if_winsor,
                       winsor=winsor)
        return ratio

    results = Parallel(n_jobs=4)(delayed(process_stock_parallel)(stock) for stock in df_op['S_INFO_WINDCODE'].unique())
    df_ration = pd.concat(results, axis=1, join='outer')
    df_ration.index = pd.to_datetime(df_ration.index.astype(str))
    df_ration.index.name = 'trade_date'
    ration_list = sorted(df_ration.index)
    unv_list = sorted(universe.index)
    union_list = list(set(ration_list).union(set(unv_list)))
    union_list = sorted(union_list)
    f = df_ration.reindex(index=union_list)
    for i in union_list:
        if i in unv_list:
            continue
        index = union_list.index(i)
        for j in union_list[index + 1:]:
            if j not in unv_list:
                continue
            f.loc[j] = f.loc[j].fillna(f.loc[i])
            break
    f = f.reindex(index=universe.index)
    f = f.dropna(how='all')
    return f





