import pandas as pd


def analyze_top_products(df, sales_col, product_col):
    """Анализ наиболее продаваемых товаров."""
    if not sales_col or not product_col:
        return {
            "answer": "Для анализа товаров нужны столбцы с продажами и с названием товара. Пожалуйста, убедитесь, что в данных есть такие столбцы.",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

    try:
        top_products = (
            df.groupby(product_col)[sales_col].agg(["sum", "count", "mean"]).round(2)
        )
        top_products = top_products.sort_values("sum", ascending=False).head(10)

        analysis = "📊 **АНАЛИЗ ЛУЧШЕ ПРОДАЮЩИХСЯ ТОВАРОВ**\n\n"
        analysis += f"На основе ваших данных о продажах с {len(df)} транзакциями:\n\n"

        for i, (product, data) in enumerate(top_products.iterrows(), 1):
            analysis += f"{i}. **{product}**\n"
            analysis += f"   • Общий объём продаж: ${data['sum']:,.2f}\n"
            analysis += f"   • Количество заказов: {data['count']}\n"
            analysis += f"   • Средний чек по товару: ${data['mean']:,.2f}\n\n"

        total_revenue = df[sales_col].sum()
        top_10_revenue = top_products["sum"].sum()
        percentage = (top_10_revenue / total_revenue) * 100

        analysis += "💡 **КЛЮЧЕВЫЕ ВЫВОДЫ:**\n"
        analysis += (
            f"• Топ‑10 товаров дают ${top_10_revenue:,.2f} ({percentage:.1f}%) общей выручки\n"
        )
        analysis += f"• Суммарная выручка по всем товарам: ${total_revenue:,.2f}\n"
        analysis += (
            f"• Лучший товар по выручке: {top_products.index[0]} с объёмом "
            f"${top_products.iloc[0]['sum']:,.2f}"
        )

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при анализе товаров: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_top_customers(df, sales_col, customer_col):
    """Анализ лучших клиентов по объёму покупок."""
    if not sales_col or not customer_col:
        return {
            "answer": "Для анализа клиентов нужны столбцы с продажами и с идентификатором/именем клиента.",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

    try:
        top_customers = (
            df.groupby(customer_col)[sales_col]
            .agg(["sum", "count", "mean"])
            .round(2)
        )
        top_customers = top_customers.sort_values("sum", ascending=False).head(10)

        analysis = "👥 **АНАЛИЗ ЛУЧШИХ КЛИЕНТОВ**\n\n"

        for i, (customer, data) in enumerate(top_customers.iterrows(), 1):
            analysis += f"{i}. **{customer}**\n"
            analysis += f"   • Всего потратил: ${data['sum']:,.2f}\n"
            analysis += f"   • Количество заказов: {data['count']}\n"
            analysis += f"   • Средний чек: ${data['mean']:,.2f}\n\n"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при анализе клиентов: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_top_sales(df, sales_col, date_col):
    """Анализ периодов с наибольшими продажами."""
    if not sales_col:
        return {
            "answer": "Для анализа максимальных продаж нужен столбец с суммой/объёмом продаж.",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

    try:
        top_sales = df.nlargest(10, sales_col)
        analysis = "💰 **ТОП ТРАНЗАКЦИЙ ПО ПРОДАЖАМ**\n\n"

        for i, (_, row) in enumerate(top_sales.iterrows(), 1):
            analysis += f"{i}. ${row[sales_col]:,.2f}"
            if date_col and date_col in row:
                analysis += f" на дату {row[date_col]}"
            analysis += "\n"

        total = df[sales_col].sum()
        avg = df[sales_col].mean()
        analysis += "\n📊 **СВОДКА:**\n"
        analysis += f"• Общий объём продаж: ${total:,.2f}\n"
        analysis += f"• Средний размер продажи: ${avg:,.2f}\n"
        analysis += f"• Максимальная разовая продажа: ${df[sales_col].max():,.2f}"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при анализе максимальных продаж: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_trends(df, sales_col, date_col):
    """Анализ трендов продаж во времени."""
    if not sales_col or not date_col:
        return {
            "answer": "Для анализа динамики во времени нужны столбцы с датой и с продажами.",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        monthly = df_copy.groupby(df_copy[date_col].dt.to_period("M"))[sales_col].sum()

        analysis = "📈 **АНАЛИЗ ТРЕНДОВ ПРОДАЖ**\n\n"
        analysis += "**Месячная динамика продаж:**\n"

        for period, sales in monthly.tail(12).items():
            analysis += f"• {period}: ${sales:,.2f}\n"

        if len(monthly) > 1:
            recent_growth = (
                (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]
            ) * 100
            analysis += "\n📊 **Анализ роста:**\n"
            analysis += f"• Рост месяц к месяцу: {recent_growth:+.1f}%\n"
            analysis += f"• Лучший месяц: {monthly.idxmax()} (${monthly.max():,.2f})\n"
            analysis += f"• Худший месяц: {monthly.idxmin()} (${monthly.min():,.2f})"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при анализе трендов: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_totals(df, sales_col):
    """Анализ общей выручки и итоговых продаж."""
    if not sales_col:
        return {
            "answer": "Для расчёта итогов нужен столбец с продажами.",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

    try:
        total_sales = df[sales_col].sum()
        total_transactions = len(df)

        analysis = "💰 **АНАЛИЗ ИТОГОВЫХ ПРОДАЖ**\n\n"
        analysis += f"• **Суммарная выручка:** ${total_sales:,.2f}\n"
        analysis += f"• **Количество транзакций:** {total_transactions:,}\n"
        analysis += f"• **Средний чек:** ${total_sales / total_transactions:,.2f}\n"
        analysis += f"• **Максимальная продажа:** ${df[sales_col].max():,.2f}\n"
        analysis += f"• **Минимальная продажа:** ${df[sales_col].min():,.2f}"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при расчёте итогов: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_averages(df, sales_col):
    """Анализ средних показателей продаж."""
    if not sales_col:
        return {
            "answer": "Для расчёта средних значений нужен столбец с продажами.",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

    try:
        mean_sale = df[sales_col].mean()
        median_sale = df[sales_col].median()
        std_dev = df[sales_col].std()

        analysis = "📊 **АНАЛИЗ СРЕДНИХ ПОКАЗАТЕЛЕЙ ПРОДАЖ**\n\n"
        analysis += f"• **Среднее значение:** ${mean_sale:,.2f}\n"
        analysis += f"• **Медиана:** ${median_sale:,.2f}\n"
        analysis += f"• **Стандартное отклонение:** ${std_dev:,.2f}\n"

        if mean_sale > median_sale:
            analysis += (
                "\n💡 **Вывод:** Распределение продаж с правым перекосом — есть несколько "
                "крупных транзакций, которые тянут среднее вверх."
            )
        else:
            analysis += (
                "\n💡 **Вывод:** Распределение продаж близко к нормальному."
            )

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при расчёте средних значений: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def get_general_insights(df, sales_col, date_col, product_col, customer_col):
    """Формирование общих выводов по данным о продажах."""
    try:
        insights = "🔍 **ОБЩИЕ ВЫВОДЫ ПО ДАННЫМ О ПРОДАЖАХ**\n\n"
        insights += "**Обзор датасета:**\n"
        insights += f"• Количество записей: {len(df):,}\n"
        insights += f"• Количество столбцов: {len(df.columns)}\n"

        if sales_col:
            total_revenue = df[sales_col].sum()
            avg_sale = df[sales_col].mean()
            insights += f"• Общая выручка: ${total_revenue:,.2f}\n"
            insights += f"• Средняя продажа: ${avg_sale:,.2f}\n"

        if date_col:
            try:
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                date_range = df_temp[date_col].max() - df_temp[date_col].min()
                insights += f"• Период данных: {date_range.days} дней\n"
            except Exception:
                pass

        if product_col:
            unique_products = df[product_col].nunique()
            insights += f"• Уникальных товаров: {unique_products}\n"

        if customer_col:
            unique_customers = df[customer_col].nunique()
            insights += f"• Уникальных клиентов: {unique_customers}\n"

        insights += "\n💡 **Что вы можете у меня спрашивать:**\n"
        insights += "• «Какие товары продаются лучше всего?» — анализ ассортимента\n"
        insights += "• «Спрогнозируй продажи на следующий месяц» — прогнозирование\n"
        insights += "• «Покажи тренды продаж» — анализ динамики\n"
        insights += "• «Какова общая выручка?» — сводная статистика\n"
        insights += "• «Кто лучшие клиенты?» — анализ клиентов\n"
        insights += "• «Есть ли сезонность?» — анализ сезонных паттернов\n"
        insights += "• «Покажи ключевые метрики» — дашборд KPI\n"
        insights += "• «Какие товары проседают по продажам?» — поиск проблемных сегментов"

        return {
            "answer": insights,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при формировании общих выводов: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_seasonality(df, sales_col, date_col):
    """Анализ сезонных паттернов в данных о продажах."""
    if not sales_col or not date_col:
        return {
            "answer": "Для анализа сезонности нужны столбцы с датой и продажами.",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

    try:
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        df_copy["month"] = df_copy[date_col].dt.month
        df_copy["quarter"] = df_copy[date_col].dt.quarter
        df_copy["day_of_week"] = df_copy[date_col].dt.day_name()

        monthly_sales = df_copy.groupby("month")[sales_col].mean().round(2)
        quarterly_sales = df_copy.groupby("quarter")[sales_col].mean().round(2)
        weekly_sales = df_copy.groupby("day_of_week")[sales_col].mean().round(2)

        analysis = "🌟 **АНАЛИЗ СЕЗОННОСТИ**\n\n"

        analysis += "**Месячные паттерны:**\n"
        month_names = [
            "Янв",
            "Фев",
            "Мар",
            "Апр",
            "Май",
            "Июн",
            "Июл",
            "Авг",
            "Сен",
            "Окт",
            "Ноя",
            "Дек",
        ]
        for month, sales in monthly_sales.items():
            analysis += f"• {month_names[month-1]}: ${sales:,.2f} в среднем\n"

        analysis += "\n**Квартальные паттерны:**\n"
        quarter_names = ["Q1", "Q2", "Q3", "Q4"]
        for quarter, sales in quarterly_sales.items():
            analysis += f"• {quarter_names[quarter-1]}: ${sales:,.2f} в среднем\n"

        analysis += "\n**Недельные паттерны:**\n"
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        day_ru = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        for i, day in enumerate(day_order):
            if day in weekly_sales.index:
                analysis += f"• {day_ru[i]}: ${weekly_sales[day]:,.2f} в среднем\n"

        best_month = monthly_sales.idxmax()
        best_quarter = quarterly_sales.idxmax()
        best_day = weekly_sales.idxmax()

        analysis += "\n💡 **Ключевые выводы:**\n"
        analysis += (
            f"• Лучший месяц: {month_names[best_month-1]} "
            f"(${monthly_sales[best_month]:,.2f})\n"
        )
        analysis += (
            f"• Лучший квартал: {quarter_names[best_quarter-1]} "
            f"(${quarterly_sales[best_quarter]:,.2f})\n"
        )
        best_day_idx = day_order.index(best_day) if best_day in day_order else 0
        analysis += (
            f"• Лучший день недели: {day_ru[best_day_idx]} "
            f"(${weekly_sales[best_day]:,.2f})\n"
        )

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при анализе сезонности: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_performance_metrics(df, sales_col, date_col, product_col, customer_col):
    """Анализ комплексных метрик эффективности (KPI-дашборд)."""
    try:
        analysis = "## 📊 Дашборд KPI\n\n"

        if sales_col:
            total_revenue = df[sales_col].sum()
            avg_transaction = df[sales_col].mean()
            median_transaction = df[sales_col].median()

            analysis += "### 💰 Метрики выручки\n\n"
            analysis += "| Метрика | Значение |\n"
            analysis += "|---|---:|\n"
            analysis += f"| Общая выручка | ${total_revenue:,.2f} |\n"
            analysis += f"| Средний чек | ${avg_transaction:,.2f} |\n"
            analysis += f"| Медианный чек | ${median_transaction:,.2f} |\n"
            analysis += f"| Количество транзакций | {len(df):,} |\n\n"

        if product_col:
            unique_products = df[product_col].nunique()
            top_product = (
                df.groupby(product_col)[sales_col].sum().idxmax() if sales_col else None
            )
            analysis += "### 🧺 Метрики товаров\n\n"
            analysis += "| Метрика | Значение |\n"
            analysis += "|---|---:|\n"
            analysis += f"| Всего товаров | {unique_products} |\n"
            if top_product is not None:
                analysis += f"| Лучший товар по выручке | {top_product} |\n"
            analysis += "\n"

        if customer_col:
            unique_customers = df[customer_col].nunique()
            repeat_customers = df[customer_col].value_counts()
            repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
            analysis += "### 👥 Метрики клиентов\n\n"
            analysis += "| Метрика | Значение |\n"
            analysis += "|---|---:|\n"
            analysis += f"| Всего клиентов | {unique_customers} |\n"
            analysis += f"| Доля повторных покупок | {repeat_rate:.1f}% |\n\n"

        if date_col and sales_col:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            date_range = (df_temp[date_col].max() - df_temp[date_col].min()).days
            days = max(date_range, 1)
            total_revenue = df[sales_col].sum()
            avg_daily = total_revenue / days
            analysis += "### 📅 Временные метрики\n\n"
            analysis += "| Метрика | Значение |\n"
            analysis += "|---|---:|\n"
            analysis += f"| Период данных | {date_range} дней |\n"
            analysis += f"| Среднедневная выручка | ${avg_daily:,.2f} |\n"
            analysis += "\n"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при анализе метрик: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }


def analyze_declining_performance(df, sales_col, product_col, customer_col):
    """Анализ товаров и клиентов с низкими показателями."""
    try:
        analysis = "📉 **АНАЛИЗ ПРОБЛЕМНЫХ СЕГМЕНТОВ**\n\n"

        if product_col and sales_col:
            product_sales = (
                df.groupby(product_col)[sales_col]
                .agg(["sum", "count", "mean"])
                .round(2)
            )
            worst_products = product_sales.sort_values("sum").head(5)

            analysis += "**Товары с наименьшими продажами:**\n"
            for i, (product, data) in enumerate(worst_products.iterrows(), 1):
                analysis += (
                    f"{i}. {product}: ${data['sum']:,.2f} всего\n"
                )

        if customer_col and sales_col:
            customer_sales = (
                df.groupby(customer_col)[sales_col].sum().round(2)
            )
            worst_customers = customer_sales.sort_values().head(5)

            analysis += "\n**Клиенты с наименьшими затратами:**\n"
            for i, (customer, sales) in enumerate(worst_customers.items(), 1):
                analysis += f"{i}. {customer}: ${sales:,.2f}\n"

        if sales_col:
            low_value_threshold = df[sales_col].quantile(0.25)
            low_value_count = (df[sales_col] <= low_value_threshold).sum()
            low_value_percentage = (low_value_count / len(df)) * 100

            analysis += "\n💡 **Выводы:**\n"
            analysis += (
                f"• {low_value_count} транзакций "
                f"({low_value_percentage:.1f}%) ниже ${low_value_threshold:.2f}\n"
            )
            analysis += (
                "• Рекомендуется разработать стратегии для улучшения слабых сегментов\n"
            )
            analysis += "• Сфокусироваться на апселле и удержании клиентов"

        return {
            "answer": analysis,
            "model": "Local Sales Analyzer",
            "status": "success",
        }
    except Exception as e:
        return {
            "answer": f"Ошибка при анализе проблемных сегментов: {str(e)}",
            "model": "Local Sales Analyzer",
            "status": "error",
        }

