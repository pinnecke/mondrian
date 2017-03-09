#pragma once

#define INVOKE_DMS_SINGLE_TUPLE_MATERIALIZATION(cursor)                                                                \
{                                                                                                                      \
    ResultSetQ2CustomerInfo *info = &ResultSet->CustomerInfos[cursor];                                                 \
    info->C_ID= customerTableColumnStore->C_ID[Params.CustomerTupleIds[cursor]];                                       \
    info->C_D_ID= customerTableColumnStore->C_D_ID[Params.CustomerTupleIds[cursor]];                                   \
    info->C_W_ID= customerTableColumnStore->C_W_ID[Params.CustomerTupleIds[cursor]];                                   \
    info->C_FIRST= customerTableColumnStore->C_FIRST[Params.CustomerTupleIds[cursor]];                                 \
    info->C_MIDDLE= customerTableColumnStore->C_MIDDLE[Params.CustomerTupleIds[cursor]];                               \
    info->C_LAST= customerTableColumnStore->C_LAST[Params.CustomerTupleIds[cursor]];                                   \
    info->C_STREET_1= customerTableColumnStore->C_STREET_1[Params.CustomerTupleIds[cursor]];                           \
    info->C_STREET_2= customerTableColumnStore->C_STREET_2[Params.CustomerTupleIds[cursor]];                           \
    info->C_CITY= customerTableColumnStore->C_CITY[Params.CustomerTupleIds[cursor]];                                   \
    info->C_STATE= customerTableColumnStore->C_STATE[Params.CustomerTupleIds[cursor]];                                 \
    info->C_ZIP= customerTableColumnStore->C_ZIP[Params.CustomerTupleIds[cursor]];                                     \
    info->C_PHONE= customerTableColumnStore->C_PHONE[Params.CustomerTupleIds[cursor]];                                 \
    info->C_SINCE= customerTableColumnStore->C_SINCE[Params.CustomerTupleIds[cursor]];                                 \
    info->C_CREDIT= customerTableColumnStore->C_CREDIT[Params.CustomerTupleIds[cursor]];                               \
    info->C_CREDIT_LIM= customerTableColumnStore->C_CREDIT_LIM[Params.CustomerTupleIds[cursor]];                       \
    info->C_DISCOUNT= customerTableColumnStore->C_DISCOUNT[Params.CustomerTupleIds[cursor]];                           \
    info->C_BALANCE= customerTableColumnStore->C_BALANCE[Params.CustomerTupleIds[cursor]];                             \
    info->C_YTD_PAYMENT= customerTableColumnStore->C_YTD_PAYMENT[Params.CustomerTupleIds[cursor]];                     \
    info->C_PAYMENT_CNT= customerTableColumnStore->C_PAYMENT_CNT[Params.CustomerTupleIds[cursor]];                     \
    info->C_DELIVERY_CNT= customerTableColumnStore->C_DELIVERY_CNT[Params.CustomerTupleIds[cursor]];                   \
    info->C_DATA= customerTableColumnStore->C_DATA[Params.CustomerTupleIds[cursor]];                                   \
}

#define INVOKE_NMS_SINGLE_TUPLE_MATERIALIZATION(cursor)                                                                \
{                                                                                                                      \
    ResultSetQ2CustomerInfo *info = &ResultSet->CustomerInfos[cursor];                                                 \
    info->C_ID = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_ID;                               \
    info->C_D_ID = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_D_ID;                           \
    info->C_W_ID = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_W_ID;                           \
    info->C_FIRST = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_FIRST;                         \
    info->C_MIDDLE = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_MIDDLE;                       \
    info->C_LAST = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_LAST;                           \
    info->C_STREET_1 = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_STREET_1;                   \
    info->C_STREET_2 = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_STREET_2;                   \
    info->C_CITY = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_CITY;                           \
    info->C_STATE = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_STATE;                         \
    info->C_ZIP = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_ZIP;                             \
    info->C_PHONE = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_PHONE;                         \
    info->C_SINCE = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_SINCE;                         \
    info->C_CREDIT = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_CREDIT;                       \
    info->C_CREDIT_LIM = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_CREDIT_LIM;               \
    info->C_DISCOUNT = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_DISCOUNT;                   \
    info->C_BALANCE = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_BALANCE;                     \
    info->C_YTD_PAYMENT = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_YTD_PAYMENT;             \
    info->C_PAYMENT_CNT = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_PAYMENT_CNT;             \
    info->C_DELIVERY_CNT = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_DELIVERY_CNT;           \
    info->C_DATA = customerTableColumnStore->Tuples[Params.CustomerTupleIds[cursor]].C_DATA;                           \
}
