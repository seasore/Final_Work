import { useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Card, Row, Col, Button, Spin, message, Select } from 'antd'
import { CloseOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'
import dayjs from 'dayjs'
import { predictionApi, getApiErrorMessage } from '../api'
import { activeCompareKeys, COMPARE_MODEL_KEYS } from '../predictionModels'

function toHourly24(points) {
  if (!points?.length) return []
  const slice = points.slice(0, 96)
  const out = []
  const keys = activeCompareKeys(slice[0])
  for (let h = 0; h < 24; h++) {
    const chunk = slice.slice(h * 4, h * 4 + 4)
    if (!chunk.length) break
    const m = chunk.reduce((s, p) => s + (p.main_mw || 0), 0) / chunk.length
    const b = chunk.reduce((s, p) => s + (p.baseline_mw || 0), 0) / chunk.length
    const row = { main: m, base: b }
    keys.forEach(({ key }) => {
      row[key] = chunk.reduce((s, p) => s + (Number(p[key]) || 0), 0) / chunk.length
    })
    out.push(row)
  }
  return out
}

export default function BigScreen() {
  const navigate = useNavigate()
  const [mounted, setMounted] = useState(false)
  const [loading, setLoading] = useState(true)
  const [pred, setPred] = useState([])
  const [weatherMeta, setWeatherMeta] = useState(null)
  const [predSteps, setPredSteps] = useState(96)

  useEffect(() => {
    document.body.style.overflow = 'hidden'
    setMounted(true)
    return () => {
      document.body.style.overflow = ''
    }
  }, [])

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const start = dayjs().startOf('day').toISOString()
        const res = await predictionApi.getTwoDays(start, predSteps)
        const rows = res?.data?.data
        setPred(Array.isArray(rows) ? rows : [])
        setWeatherMeta(res?.data?.weather ?? null)
        if (!Array.isArray(rows)) {
          message.warning('预测接口返回异常，请用 npm run dev 打开页面并确认后端已启动')
        }
      } catch (e) {
        message.error(getApiErrorMessage(e))
        setPred([])
        setWeatherMeta(null)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [predSteps])

  const compareKeysOnData = useMemo(() => activeCompareKeys(pred[0]), [pred])

  const hourly = useMemo(() => toHourly24(pred), [pred])

  const chartOption = useMemo(() => {
    const series = [
      {
        type: 'line',
        data: hourly.map((x) => x.main),
        name: 'TCN-BiLSTM-Attention',
        smooth: true,
        lineStyle: { color: '#40a9ff', width: 3 },
        symbol: 'circle',
        symbolSize: 4,
      },
    ]
    compareKeysOnData.forEach(({ key, short, color }) => {
      series.push({
        type: 'line',
        data: hourly.map((x) => x[key] ?? null),
        name: short,
        smooth: true,
        lineStyle: { color, width: 1.5 },
        symbol: 'none',
      })
    })
    series.push({
      type: 'line',
      data: hourly.map((x) => x.base),
      name: '持久化模型',
      smooth: true,
      lineStyle: { color: '#ff4d4f', width: 1.5, type: 'dashed' },
      symbol: 'none',
    })
    return {
      backgroundColor: 'transparent',
      title: { text: '首日多模型预测（按小时聚合）', textStyle: { color: '#e8f4ff', fontSize: 16 }, left: 'center', top: 4 },
      grid: {
        containLabel: true,
        left: 12,
        right: 18,
        top: 52,
        bottom: 18,
      },
      xAxis: { type: 'category', data: hourly.map((_, i) => `${i}时`), axisLabel: { color: '#e8f4ff' } },
      yAxis: {
        type: 'value',
        name: 'MW',
        axisLabel: { color: '#e8f4ff' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      },
      series,
      legend: {
        type: 'scroll',
        textStyle: { color: '#e8f4ff' },
        top: 28,
      },
    }
  }, [hourly, compareKeysOnData])

  const dayAvg = useMemo(() => {
    if (!pred.length) return { main: 0, base: 0, byKey: {} }
    const day1 = pred.slice(0, 96)
    const n = day1.length || 1
    const meanKey = (k) => day1.reduce((s, p) => s + (Number(p[k]) || 0), 0) / n
    const byKey = {}
    COMPARE_MODEL_KEYS.forEach(({ key }) => {
      if (day1[0] && day1[0][key] != null && !Number.isNaN(Number(day1[0][key]))) {
        byKey[key] = meanKey(key)
      }
    })
    return { main: meanKey('main_mw'), base: meanKey('baseline_mw'), byKey }
  }, [pred])

  const rankOption = useMemo(() => {
    const items = [{ name: '持久化模型', val: dayAvg.base, color: '#8c8c8c' }]
    compareKeysOnData.forEach(({ key, short, color }) => {
      const v = dayAvg.byKey[key]
      if (v != null) items.push({ name: short, val: v, color })
    })
    items.push({ name: '本文主模型', val: dayAvg.main, color: '#40a9ff' })
    const sorted = [...items].sort((a, b) => a.val - b.val)
    return {
      backgroundColor: 'transparent',
      title: { text: '首日预测平均功率对比（MW）', textStyle: { color: '#e8f4ff', fontSize: 16 }, left: 'center', top: 4 },
      grid: {
        containLabel: true,
        left: 12,
        right: 20,
        top: 52,
        bottom: 16,
      },
      xAxis: { type: 'value', axisLabel: { color: '#e8f4ff' } },
      yAxis: {
        type: 'category',
        data: sorted.map((x) => x.name),
        axisLabel: {
          color: '#e8f4ff',
          fontSize: 12,
          margin: 10,
          hideOverlap: false,
          align: 'right',
          verticalAlign: 'middle',
        },
        axisTick: { alignWithLabel: true },
      },
      series: [
        {
          type: 'bar',
          data: sorted.map((x) => ({ value: x.val, itemStyle: { color: x.color } })),
        },
      ],
    }
  }, [dayAvg, compareKeysOnData])

  const accMain = pred.length
    ? Math.max(0, 100 - Math.min(30, Math.abs(dayAvg.main - dayAvg.base) / (dayAvg.main || 1) * 5))
    : 0
  const accBase = pred.length
    ? Math.max(0, 100 - Math.min(35, Math.abs(dayAvg.main - dayAvg.base) / (dayAvg.base || 1) * 8))
    : 0

  const wx = weatherMeta?.forecast?.[0]
  const cardStyle = {
    background: 'rgba(255,255,255,0.06)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: 12,
    boxShadow: '0 8px 32px rgba(0,0,0,0.2)',
    padding: 20,
    transition: 'all 0.3s',
  }

  if (!mounted) return null
  return (
    <div
      className="main-bg"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        padding: 24,
        overflow: 'auto',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ color: '#e8f4ff', margin: 0 }}>短期负荷预测 - 大屏概览</h1>
          {weatherMeta?.location_label && (
            <div style={{ color: 'rgba(232,244,255,0.65)', fontSize: 13, marginTop: 8 }}>
              气象区域：{weatherMeta.location_label}
              {weatherMeta.source === 'openweather' && '（OpenWeatherMap 实况预报）'}
              {weatherMeta.source === 'training_climatology' && '（训练集气候学回退）'}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: '#e8f4ff', fontSize: 13 }}>预测长度</span>
          <Select
            value={predSteps}
            onChange={setPredSteps}
            style={{ width: 200 }}
            options={[
              { value: 96, label: '快速 96 点' },
              { value: 192, label: '完整 192 点' },
            ]}
          />
          <Button type="primary" icon={<CloseOutlined />} onClick={() => navigate('/dashboard')} className="tech-btn">
            退出大屏模式
          </Button>
        </div>
      </div>
      <Spin spinning={loading}>
        <Row gutter={24}>
          <Col xs={24} lg={15}>
            <div
              style={cardStyle}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = '0 12px 40px rgba(24,144,255,0.25)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.2)'
              }}
            >
              <ReactECharts option={chartOption} style={{ height: 350 }} notMerge />
            </div>
          </Col>
          <Col xs={24} lg={9}>
            <div
              style={cardStyle}
              onMouseEnter={(e) => {
                e.currentTarget.style.boxShadow = '0 12px 40px rgba(24,144,255,0.25)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.boxShadow = '0 8px 32px rgba(0,0,0,0.2)'
              }}
            >
              <ReactECharts option={rankOption} style={{ height: 350 }} notMerge />
            </div>
          </Col>
        </Row>
        <Row gutter={24} style={{ marginTop: 24 }}>
          <Col span={6}>
            <div style={cardStyle}>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: 14 }}>首日主模型预测均值</div>
              <div style={{ color: '#40a9ff', fontSize: 28, fontWeight: 600 }}>
                {dayAvg.main ? `${dayAvg.main.toFixed(1)} MW` : '—'}
              </div>
            </div>
          </Col>
          <Col span={6}>
            <div style={cardStyle}>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: 14 }}>主模型稳定性指数</div>
              <div style={{ color: '#40a9ff', fontSize: 28, fontWeight: 600 }}>
                {pred.length ? `${accMain.toFixed(1)}%` : '—'}
              </div>
            </div>
          </Col>
          <Col span={6}>
            <div style={cardStyle}>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: 14 }}>持久化基线稳定性指数</div>
              <div style={{ color: '#ff4d4f', fontSize: 28, fontWeight: 600 }}>
                {pred.length ? `${accBase.toFixed(1)}%` : '—'}
              </div>
            </div>
          </Col>
          <Col span={6}>
            <div style={cardStyle}>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: 14 }}>气象预警</div>
              <div style={{ color: '#52c41a', fontSize: 16 }}>
                {wx?.weather_main === 'Rain' || wx?.weather_main === 'Thunderstorm' ? '请关注降水' : '无'}
              </div>
            </div>
          </Col>
        </Row>
        <Row gutter={24} style={{ marginTop: 24 }}>
          <Col span={24}>
            <div style={cardStyle}>
              <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: 14, marginBottom: 12 }}>
                气象信息（{weatherMeta?.source === 'openweather' ? 'API' : weatherMeta?.source === 'training_climatology' ? '气候学估计' : '—'}）
              </div>
              {wx ? (
                <Row gutter={24}>
                  <Col span={4}>
                    <div style={{ color: '#e8f4ff' }}>概况</div>
                    <div style={{ color: '#fff', fontSize: 18 }}>{wx.description || '—'}</div>
                  </Col>
                  <Col span={4}>
                    <div style={{ color: '#e8f4ff' }}>湿度</div>
                    <div style={{ color: '#fff', fontSize: 18 }}>
                      {wx.humidity != null ? `${Number(wx.humidity).toFixed(0)}%` : '—'}
                    </div>
                  </Col>
                  <Col span={4}>
                    <div style={{ color: '#e8f4ff' }}>最高温度</div>
                    <div style={{ color: '#fff', fontSize: 18 }}>
                      {wx.max_temp != null ? `${Number(wx.max_temp).toFixed(1)}°C` : '—'}
                    </div>
                  </Col>
                  <Col span={4}>
                    <div style={{ color: '#e8f4ff' }}>最低温度</div>
                    <div style={{ color: '#fff', fontSize: 18 }}>
                      {wx.min_temp != null ? `${Number(wx.min_temp).toFixed(1)}°C` : '—'}
                    </div>
                  </Col>
                  <Col span={4}>
                    <div style={{ color: '#e8f4ff' }}>风速</div>
                    <div style={{ color: '#fff', fontSize: 18 }}>
                      {wx.wind_speed != null ? `${Number(wx.wind_speed).toFixed(1)} m/s` : '—'}
                    </div>
                  </Col>
                  <Col span={4}>
                    <div style={{ color: '#e8f4ff' }}>气压</div>
                    <div style={{ color: '#fff', fontSize: 18 }}>
                      {wx.pressure != null ? `${Number(wx.pressure).toFixed(0)} hPa` : '—'}
                    </div>
                  </Col>
                </Row>
              ) : (
                <div style={{ color: 'rgba(255,255,255,0.5)' }}>暂无气象数据（请检查后端 WEATHER_API_KEY 与训练集路径）</div>
              )}
              {weatherMeta?.error && (
                <div style={{ color: '#faad14', fontSize: 12, marginTop: 12 }}>{weatherMeta.error}</div>
              )}
            </div>
          </Col>
        </Row>
      </Spin>
    </div>
  )
}
