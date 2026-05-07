import { useEffect, useState, useCallback } from 'react'
import { Card, Table, Button, Upload, Tabs, message, Spin, Alert } from 'antd'
import { UploadOutlined } from '@ant-design/icons'
import { weatherApi, getApiErrorMessage } from '../api'

export default function DataManage() {
  const [wxLoading, setWxLoading] = useState(true)
  const [wxRows, setWxRows] = useState([])
  const [wxError, setWxError] = useState('')

  const loadWeather = useCallback(async () => {
    setWxLoading(true)
    setWxError('')
    try {
      const res = await weatherApi.getForecast(2)
      const w = res?.data
      if (!w || typeof w !== 'object') {
        setWxRows([])
        setWxError('接口返回异常，请确认通过 npm run dev 访问且后端已启动。')
        return
      }
      const label = w.location_label || w.city_query || ''
      const rows = (w.forecast || []).map((d, i) => ({
        key: i,
        unit: label,
        date: d.date || '—',
        temp: d.max_temp != null && d.min_temp != null
          ? `${Number(d.min_temp).toFixed(1)}～${Number(d.max_temp).toFixed(1)}`
          : '—',
        humidity: d.humidity != null ? Number(d.humidity).toFixed(0) : '—',
        wind: d.wind_speed != null ? Number(d.wind_speed).toFixed(1) : '—',
        desc: d.description || '—',
      }))
      setWxRows(rows)
      if (!rows.length) {
        setWxError(w.error || '预报列表为空，请检查后端日志与训练集路径。')
      }
    } catch (e) {
      setWxRows([])
      setWxError(getApiErrorMessage(e))
      message.error(getApiErrorMessage(e))
    } finally {
      setWxLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadWeather()
  }, [loadWeather])

  const loadData = []

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <Upload><Button icon={<UploadOutlined />} className="tech-btn" style={{ flexShrink: 0 }}>批量导入 Excel/CSV</Button></Upload>
          <Button style={{ flexShrink: 0 }} onClick={() => message.info('手动录入需对接后端存储接口')}>手动录入</Button>
          <Button style={{ flexShrink: 0 }} onClick={() => { const csv = '单位,日期,最大负荷,最小负荷,平均\n'; const a = document.createElement('a'); a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent('\ufeff' + csv); a.download = '负荷数据.csv'; a.click(); message.success('已导出表头模板'); }}>批量导出</Button>
          <Button onClick={() => void loadWeather()} loading={wxLoading}>刷新气象</Button>
        </div>
      </Card>
      {wxError && (
        <Alert type="warning" showIcon message={wxError} style={{ marginBottom: 16 }} />
      )}
      <Card className="glass-card">
        <Tabs
          items={[
            {
              key: 'load',
              label: '负荷数据',
              children: (
                <Table
                  locale={{ emptyText: '暂无本地录入数据（导入/录入功能需对接后端数据管理 API）' }}
                  dataSource={loadData}
                  columns={[
                    { title: '单位', dataIndex: 'unit' },
                    { title: '日期', dataIndex: 'date' },
                    { title: '最大负荷', dataIndex: 'max' },
                    { title: '最小负荷', dataIndex: 'min' },
                    { title: '平均负荷', dataIndex: 'avg' },
                    { title: '操作', render: () => <><Button type="link" size="small">修改</Button><Button type="link" size="small" danger>删除</Button></> },
                  ]}
                />
              ),
            },
            {
              key: 'weather',
              label: '气象数据',
              children: (
                <Spin spinning={wxLoading}>
                  <Table
                    locale={{ emptyText: wxError || '暂无气象数据' }}
                    dataSource={wxRows}
                    columns={[
                      { title: '区域', dataIndex: 'unit', width: 200 },
                      { title: '日期', dataIndex: 'date' },
                      { title: '温度 (°C)', dataIndex: 'temp' },
                      { title: '湿度 (%)', dataIndex: 'humidity' },
                      { title: '风速 (m/s)', dataIndex: 'wind' },
                      { title: '概况', dataIndex: 'desc' },
                    ]}
                  />
                </Spin>
              ),
            },
          ]}
        />
      </Card>
    </div>
  )
}
