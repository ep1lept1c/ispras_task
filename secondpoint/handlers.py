from aiogram import types, F, Router
from aiogram.types import Message
from aiogram.filters import Command, StateFilter
from aiogram.utils.formatting import Text, Bold
from aiogram import Router
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import State, StatesGroup
from typing import Callable, Dict, Any, Awaitable
from aiogram.types import TelegramObject
from utils import predict_single
from aiogram import flags
import text as text
import kb as kb
router = Router()

class AddDescription(StatesGroup):
    adding_description = State()
    choosing_truefalse = State()
    adding_category = State()

@router.message(Command("start"))
async def start_hndl(msg: Message):
    await msg.answer(text.hello.format(name=msg.from_user.full_name), reply_markup=kb.menu)

@router.message(Command("menu"))
@router.message(F.text == "Меню")
@router.message(F.text == "Выйти в меню")
@router.message(F.text == "◀️ Выйти в меню")
async def menu_hndl(msg: Message):
    await msg.answer(text.menu, reply_markup=kb.menu)

@router.callback_query(F.data == "menu")
async def menu_hndl(callback: types.CallbackQuery):
    await callback.message.answer(text.menu, reply_markup=kb.menu)
    
@router.callback_query(F.data == "add_description")
async def state_add_description_hndl(msg: Message, state: FSMContext):
    await msg.answer(text.description)
    await state.set_state(AddDescription.adding_description)


@router.message(AddDescription.adding_description)
@flags.chat_action("typing")
async def add_description(msg: Message, state: FSMContext):
    current_data = msg.text
    prediction, important_text = predict_single(current_data)
    importantdata = await state.get_data()
    important_list = importantdata.get('important', [])
    important_list.append(important_text)
    await state.update_data(important=important_list)
    await msg.answer(text.added.format(classname=prediction), reply_markup=kb.askmenu)
    await state.set_state(AddDescription.choosing_truefalse)
    
   

@router.callback_query(AddDescription.choosing_truefalse, F.data == "good_classification")
async def nice_category(callback: types.CallbackQuery, state: FSMContext):
    importantdata = await state.get_data()
    important_list = importantdata.get('important', [])
    if len(important_list[0]):
        important_text = ", ".join(important_list[0])
        await callback.message.answer(f'Боту помогли эти слова для определения класса: {important_text}\n{text.endofconv}', reply_markup=kb.iexit_kb)
    else:
        await callback.message.answer(f'Такой класс мы не рассматриваем в качестве продажи.\n {text.endofconv}', reply_markup=kb.iexit_kb)
    await state.clear()


@router.callback_query(AddDescription.choosing_truefalse, F.data == "bad_classification")
async def bad_category(callback: types.CallbackQuery, state: FSMContext):
    await callback.message.answer(text.classify_category, reply_markup=kb.category)
    await state.set_state(AddDescription.adding_category)


@router.message(AddDescription.adding_category, F.text.in_(text.available_categories))
async def add_category(msg: Message, state: FSMContext):
    await msg.answer(f'Вы выбрали категорию {msg.text}. {text.endofconv}', reply_markup=kb.iexit_kb)
    await state.clear()

@router.message(AddDescription.adding_category)
async def add_category_incorrectly(msg: Message, state: FSMContext):
    await msg.answer(f'{text.not_in_classes}', reply_markup=kb.category)
    await state.set_state(AddDescription.adding_category)
