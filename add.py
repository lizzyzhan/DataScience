class Report():
    '''
    报告自动生成工具
    r=Report(filename='')
    r.add_cover(title='reportgen')
    r.add_slides([])
    r.save()
    '''
    def __init__(self,filename=None,chart_type_default='COLUMN_CLUSTERED'):       

        self.template=template
        self.chart_type_default=chart_type_default
        if filename is None:
            if os.path.exists('template.pptx'):
                prs=Presentation('template.pptx')
            else:
                prs=Presentation()
        else :
            prs=Presentation(filename)
        self.prs=prs
        title_only_slide=pptx_layouts(prs)
        if title_only_slide:
            layouts=title_only_slide[0]
        else:
            layouts=[0,0]
        self.layouts_default=layouts
        
    def add_slides(self,slides_data):
        slides_data=slides_data_gen(slides_data)
        for slide in slides_data:
            slide_type=slide['slide_type']
            title=slide['title']
            summary=slide['summary']
            footnote=slide['footnote']
            layouts=self.layouts_default if slide['layouts'] == 'auto' else slide['layouts']
            data=slide['data']
            chart_type=slide['chart_type']
            data_config=slide['data_config']
            if (slide_type is None) or (not isinstance(slide_type,str)):
                continue
            if slide_type == 'chart':
                plot_chart(self.prs,data,chart_type=data_config,title=title,summary=summary,layouts=layouts);
            elif slide_type == 'table':
                plot_table(self.prs,data,layouts=layouts,title=title,summary=summary);
            elif slide_type in ['text','txt']:
                plot_textbox(self.prs,data,layouts=layouts,title=title,summary=summary);
            elif slide_type in ['picture','figure']:
                plot_picture(self.prs,data,layouts=layouts,title=title,summary=summary,footnote=footnote);
        return self
