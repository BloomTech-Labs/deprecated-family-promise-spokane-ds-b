"""Data visualization functions
   """

from fastapi import APIRouter
from fps_dashboard import des_statistics, ml_interpretations

router = APIRouter()


@router.post('/draw_pop_chart')
async def bar_chart(popChart):
   return {'Bar Chart': des_statistics.drawPopBarchart(populationBarChart)}


@router.post('/draw_exit_comparison')
def exit_comparison_fact_chart(exitFacetChart):
   return {'Exit Comparison': des_statistics.drawExitComparisonFacetChart(exitFacetChart)}


@router.post('/make_class_metrics')
async def metrics(target, pred, training_set, model, ml_name):
   class_metrics = ml_interpretations.make_class_metrics(target, pred, training_set, model, ml_name)
   return {'Class Metrics': class_metrics}


@router.post('/show_eli5')
def eli5_interpretation(training_set, target, model, features, X, ml_name):
   """to display most important features via permutation in eli5 and sklearn formats"""
   eli5_disp = ml_interpretations.make_eli5_interpretation(training_set, target, model, features, X, ml_name)   
   return {'ELI5 Display': eli5_disp}


@router.post('/show_PDP')
def pdp_interpretation(dataset, column_names, training_set, model):
   """to display partial dependence plots based on user input"""
   pdp_show = ml_interpretations.make_pdp_interpretation(dataset, column_names, training_set, model)
   return {'PDP': pdp_show}


@router.post('/shap_show')
def shap_interpretation(model, training_set, column_names, ml_name, target, dataset, X, processor):
   shap_show = ml_interpretations.make_shap_interpretation(model, training_set, column_names, ml_name, target, dataset, X, processor)
