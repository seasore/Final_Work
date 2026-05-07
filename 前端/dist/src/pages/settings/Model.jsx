import { Card, Form, InputNumber, Button, Tabs } from 'antd'

export default function SettingsModel() {
  return (
    <Card className="glass-card" title="算法参数设置">
      <Tabs
        items={[
          { key: 'main', label: '主模型参数', children: <Form layout="vertical"><Form.Item name="lr" label="学习率"><InputNumber style={{ width: 200 }} /></Form.Item><Form.Item name="hidden" label="隐藏层维度"><InputNumber style={{ width: 200 }} /></Form.Item><Button type="primary" className="tech-btn">保存</Button></Form> },
          { key: 'base', label: '基线模型参数', children: <Form layout="vertical"><Form.Item name="lr" label="学习率"><InputNumber style={{ width: 200 }} /></Form.Item><Button type="primary" className="tech-btn">保存</Button></Form> },
        ]}
      />
    </Card>
  )
}
