from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
menu = [
    [InlineKeyboardButton(
        text="📝 Добавить текст-описание товара", callback_data="add_description")]
]
menu = InlineKeyboardMarkup(inline_keyboard=menu)
exit_kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="◀️ Выйти в меню")]], resize_keyboard=True)
iexit_kb = InlineKeyboardMarkup(inline_keyboard=[
                                [InlineKeyboardButton(text="◀️ Выйти в меню", callback_data="menu")]])
askmenu = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Да, верно", callback_data="good_classification")],
                                                [InlineKeyboardButton(text="Нет, неверно", callback_data="bad_classification")]])
categories_list = [
    [KeyboardButton(text="Household")],
    [KeyboardButton(text="Books")],
    [KeyboardButton(text="Clothing & Accessories")],
    [KeyboardButton(text="Electronics")],
    [KeyboardButton(text="Other")]
]
category = ReplyKeyboardMarkup(keyboard=categories_list, resize_keyboard=True)
