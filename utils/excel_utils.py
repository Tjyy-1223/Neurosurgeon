import xlrd
import xlwt
from xlutils.copy import copy


""" 设计一个函数，将结果保存在excel表中，excel的名字需要自己能够命名 """
def create_excel_xsl(path, sheet_name, value):
    index = len(value)
    try:
        with xlrd.open_workbook(path) as workbook:
            workbook = copy(workbook)
            # worksheet = workbook.sheet_by_name(sheet_name)
            worksheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
            for i in range(len(value[0])):
                worksheet.col(i).width = 256 * 30  # Set the column width
            for i in range(0, index):
                for j in range(0, len(value[i])):
                    worksheet.write(i, j, value[i][j])
            workbook.save(path)
            print("xls格式表格创建成功")
    except FileNotFoundError:
        workbook = xlwt.Workbook()  # 新建一个工作簿
        worksheet = workbook.add_sheet(sheet_name)  # 在工作簿中新建一个表格
        for i in range(len(value[0])):
            worksheet.col(i).width = 256 * 30  # Set the column width
        for i in range(0, index):
            for j in range(0, len(value[i])):
                worksheet.write(i, j, value[i][j])
        workbook.save(path)
        print("xls格式表格创建成功")


""" 向excel表中写入数据"""
def write_excel_xls_append(path, sheet_name, value):
    index = len(value)
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_name(sheet_name)

    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(sheet_name)

    for i in range(len(value[0])):
        new_worksheet.col(i).width = 256 * 30  # Set the column width

    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i + rows_old, j, value[i][j])

    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")


def sheet_exists(path, sheet_name):
    try:
        workbook = xlrd.open_workbook(path)
        worksheet = workbook.sheet_by_name(sheet_name)
        if worksheet is None:
            return False
    except Exception:
        return False

""" 读取excel表格中的数据 """
def read_excel_xls(path, sheet_name):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    worksheet = workbook.sheet_by_name(sheet_name)  # 获取工作簿中的所有表格
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
        print()


""" 读取指定列中的数据 param:col_name """
def get_excel_data(path, sheet_name, col_name):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    worksheet = workbook.sheet_by_name(sheet_name)  # 获取工作簿中的所有表格

    col_index = -1
    for j in range(0, worksheet.ncols):
        if worksheet.cell_value(0, j) == col_name:
            col_index = j
    if col_index == -1:
        print("no matched col name")
        return None

    """
        开始取相应列的数据
    """
    data = []
    for i in range(1, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            if j == col_index:
                data.append(worksheet.cell_value(i, j))
    return data