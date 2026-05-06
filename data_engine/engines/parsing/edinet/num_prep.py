import numpy as np
import pandas as pd


def preproc_num(org_data:pd.DataFrame)->pd.DataFrame:
    org_data['data'] = pd.to_numeric(org_data['data_str'], errors='coerce')
    org_data['data'] = org_data['data']
    org_data['data'] = org_data.data.astype('float') # for case of data is string object (cannot read csv with dtype=int)
    org_data = org_data.assign(
        context_ref_len=(
            org_data.context_ref
            .str.split('_',expand=True)
            .notna().sum(axis=1) # count component separated by '_'
            ),
        )
    return org_data

def fill_df(data_df):
    cross_index = pd.DataFrame(data_df.docid.drop_duplicates()).join(pd.DataFrame(data_df.key.drop_duplicates()),how='cross')
    merged = pd.merge(
        cross_index,
        data_df,
        left_on=['docid','key'],
        right_on=['docid','key'],
        how='left'
    )
    merged = merged.sort_values(by=['docid','data'])
    merged = merged.assign(
        decimals=merged.decimals.ffill(),
        precision=merged.precision.ffill(),
        context_ref=merged.context_ref.ffill(),
        unit=merged.unit.ffill(),
        period_start=merged.period_start.ffill(),
        period_end=merged.period_end.ffill(),
        instant_date=merged.instant_date.ffill(),
        order=merged.order.ffill(),
        non_consolidated_flg=merged.non_consolidated_flg.ffill(),
        current_flg=merged.current_flg.ffill(),
        prior_flg=merged.prior_flg.ffill(),
        role=merged.role.ffill(),
        filerName=merged.filerName.ffill(),
        sector_label_33=merged.sector_label_33.ffill()

    )
    merged = merged.sort_values(by=['key','data'])
    merged = merged.assign(
        element_name=merged.element_name.ffill(),
        isTextBlock_flg=merged.isTextBlock_flg.ffill(),
        abstract_flg=merged.abstract_flg.ffill(),
        label_jp=merged.label_jp.ffill()
    )
    merged = merged.sort_index()
    merged.data = merged.data.fillna(-1)
    merged.data = merged.data.replace(0,np.nan).fillna(0.5*10**(merged['decimals'].astype(float)*-1))
    merged.data = merged.data.replace(-1,0)
    return merged