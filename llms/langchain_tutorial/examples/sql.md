## 创建表

```sql
# 分区表
create table test_t2(words string,frequency string) partitioned by (partdate string) row format delimited fields terminated by ',';

# orc表
CREATE TABLE IF NOT EXISTS bank.account_orc (
  `id_card` int,
  `tran_time` string,
  `name` string,
  `cash` int
  )
stored as orc;
```

# 插入数据

```sql
insert into tablename values('col1', 'col2');


INSERT INTO table_name (column1, column2, column3)
VALUES
(value1, value2, value3),
(value4, value5, value6),
(value7, value8, value9);


INSERT OVERWRITE TABLE tb
select * from tb2
;
```